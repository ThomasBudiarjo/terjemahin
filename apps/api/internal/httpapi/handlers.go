package httpapi

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/thomas/terjemahin/api/internal/config"
	"github.com/thomas/terjemahin/api/internal/jobs"
	"github.com/thomas/terjemahin/api/internal/queue"
)

const maxUpload = 25 << 20

type server struct {
	cfg   config.Config
	store jobs.Store
	q     *queue.Queue
}

func New(cfg config.Config, store jobs.Store, q *queue.Queue) http.Handler {
	s := &server{cfg: cfg, store: store, q: q}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("POST /v1/pages", s.createPage)
	mux.HandleFunc("GET /v1/jobs/{id}", s.getJob)
	mux.HandleFunc("GET /v1/jobs/{id}/image", s.getJobImage)
	mux.HandleFunc("GET /v1/jobs/{id}/result", s.getJobResult)
	return mux
}

// createPage accepts multipart form: file=<image>, context=<PageContext JSON, optional>.
func (s *server) createPage(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)
	if err := r.ParseMultipartForm(maxUpload); err != nil {
		httpError(w, http.StatusBadRequest, "invalid multipart form")
		return
	}
	file, hdr, err := r.FormFile("file")
	if err != nil {
		httpError(w, http.StatusBadRequest, "missing file")
		return
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(hdr.Filename))
	if ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp" {
		httpError(w, http.StatusBadRequest, "unsupported image type")
		return
	}

	id := newID()
	rel := filepath.Join("pages", id)
	dir := filepath.Join(s.cfg.StorageDir, rel)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		httpError(w, http.StatusInternalServerError, "storage")
		return
	}
	inputRel := filepath.Join(rel, "input"+ext)
	dst, err := os.Create(filepath.Join(s.cfg.StorageDir, inputRel))
	if err != nil {
		httpError(w, http.StatusInternalServerError, "storage")
		return
	}
	if _, err := io.Copy(dst, file); err != nil {
		dst.Close()
		httpError(w, http.StatusInternalServerError, "storage")
		return
	}
	dst.Close()

	var ctx json.RawMessage
	if c := r.FormValue("context"); c != "" {
		if !json.Valid([]byte(c)) {
			httpError(w, http.StatusBadRequest, "context must be JSON")
			return
		}
		ctx = json.RawMessage(c)
	}

	now := time.Now()
	job := jobs.Job{
		JobID:      id,
		PageID:     id,
		InputPath:  inputRel,
		OutputPath: filepath.Join(rel, "output.png"),
		ResultPath: filepath.Join(rel, "result.json"),
		Context:    ctx,
		Status:     jobs.StatusQueued,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	if err := s.q.Publish(r.Context(), job); err != nil {
		httpError(w, http.StatusServiceUnavailable, "queue unavailable")
		return
	}
	s.store.Put(job)
	writeJSON(w, http.StatusAccepted, job)
}

func (s *server) getJob(w http.ResponseWriter, r *http.Request) {
	job, ok := s.store.Get(r.PathValue("id"))
	if !ok {
		httpError(w, http.StatusNotFound, "job not found")
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (s *server) getJobImage(w http.ResponseWriter, r *http.Request) {
	s.serveArtifact(w, r, func(j jobs.Job) string { return j.OutputPath })
}

func (s *server) getJobResult(w http.ResponseWriter, r *http.Request) {
	s.serveArtifact(w, r, func(j jobs.Job) string { return j.ResultPath })
}

func (s *server) serveArtifact(w http.ResponseWriter, r *http.Request, pick func(jobs.Job) string) {
	job, ok := s.store.Get(r.PathValue("id"))
	if !ok {
		httpError(w, http.StatusNotFound, "job not found")
		return
	}
	if job.Status != jobs.StatusDone {
		httpError(w, http.StatusConflict, "job not finished: "+string(job.Status))
		return
	}
	http.ServeFile(w, r, filepath.Join(s.cfg.StorageDir, pick(job)))
}

func newID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
