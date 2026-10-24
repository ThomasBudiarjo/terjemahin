package httpapi

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/thomas/terjemahin/api/internal/config"
	"github.com/thomas/terjemahin/api/internal/jobs"
)

const maxUpload = 25 << 20

// debugStages are the per-stage images the worker writes into a job's debug dir.
var debugStages = map[string]bool{"detect": true, "mask": true, "inpainted": true, "final": true}

// Publisher queues a job for the worker (*queue.Queue in production).
type Publisher interface {
	Publish(ctx context.Context, j jobs.Job) error
}

type server struct {
	cfg   config.Config
	store jobs.Store
	q     Publisher
}

func New(cfg config.Config, store jobs.Store, q Publisher) http.Handler {
	s := &server{cfg: cfg, store: store, q: q}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("POST /v1/pages", s.createPage)
	mux.HandleFunc("GET /v1/jobs/{id}", s.getJob)
	mux.HandleFunc("GET /v1/jobs/{id}/image", s.getJobImage)
	mux.HandleFunc("GET /v1/jobs/{id}/result", s.getJobResult)
	mux.HandleFunc("GET /v1/jobs/{id}/debug/{stage}", s.getJobDebug)
	return mux
}

// pipelineConfig mirrors the worker's PipelineConfig; it is only used to validate the field,
// the raw JSON is what gets queued.
type pipelineConfig struct {
	Detector    *string                   `json:"detector"`
	OCR         *string                   `json:"ocr"`
	Translator  *string                   `json:"translator"`
	MaskRefiner *string                   `json:"mask_refiner"`
	Inpainter   *string                   `json:"inpainter"`
	Renderer    *string                   `json:"renderer"`
	SourceLang  *string                   `json:"source_lang"`
	TargetLang  *string                   `json:"target_lang"`
	Options     map[string]map[string]any `json:"options"`
	// MinOCRConfidence: OCR reads below it are left untouched (worker Settings default).
	MinOCRConfidence *float64 `json:"min_ocr_confidence"`
}

// pageContext mirrors the worker's PageContext. The worker rejects a message whose context
// does not validate, and it cannot always report that back, so the job would stay "queued"
// forever: the API checks the shape here and answers 400 instead.
type pageContext struct {
	SeriesTitle   *string          `json:"series_title"`
	SeriesSummary *string          `json:"series_summary"`
	Glossary      *[]glossaryEntry `json:"glossary"`
	PreviousPages *[][]string      `json:"previous_pages"`
	StyleNotes    *string          `json:"style_notes"`
}

type glossaryEntry struct {
	Source *string `json:"source"`
	Target *string `json:"target"`
	Note   *string `json:"note"`
}

func validateContext(raw []byte) error {
	var probe any
	if err := json.Unmarshal(raw, &probe); err != nil {
		return errors.New("context must be JSON")
	}
	if _, ok := probe.(map[string]any); !ok {
		return errors.New("context must be a JSON object")
	}
	var c pageContext
	if err := json.Unmarshal(raw, &c); err != nil {
		var te *json.UnmarshalTypeError
		if errors.As(err, &te) {
			return fmt.Errorf("context: %s must be %s, not %s", te.Field, te.Type, te.Value)
		}
		return fmt.Errorf("context: %v", err)
	}
	if c.Glossary != nil {
		for i, g := range *c.Glossary {
			if g.Source == nil || g.Target == nil {
				return fmt.Errorf("context: glossary[%d] needs source and target", i)
			}
		}
	}
	return nil
}

var pipelineStages = map[string]bool{
	"detector": true, "ocr": true, "translator": true,
	"mask_refiner": true, "inpainter": true, "renderer": true,
}

// jobOptions are the optional form fields of POST /v1/pages.
type jobOptions struct {
	Context  json.RawMessage
	Pipeline json.RawMessage
	Debug    bool
}

func parseJobOptions(r *http.Request) (jobOptions, error) {
	var o jobOptions
	if c := strings.TrimSpace(r.FormValue("context")); c != "" {
		if err := validateContext([]byte(c)); err != nil {
			return o, err
		}
		o.Context = json.RawMessage(c)
	}
	if p := strings.TrimSpace(r.FormValue("pipeline")); p != "" {
		dec := json.NewDecoder(bytes.NewReader([]byte(p)))
		dec.DisallowUnknownFields()
		var cfg pipelineConfig
		if err := dec.Decode(&cfg); err != nil {
			return o, fmt.Errorf("pipeline: %v", err)
		}
		if dec.More() {
			return o, errors.New("pipeline: trailing data after JSON object")
		}
		for stage := range cfg.Options {
			if !pipelineStages[stage] {
				return o, fmt.Errorf("pipeline: options for unknown stage %q", stage)
			}
		}
		o.Pipeline = json.RawMessage(p)
	}
	switch strings.ToLower(strings.TrimSpace(r.FormValue("debug"))) {
	case "", "0", "false", "no", "off":
	case "1", "true", "yes", "on":
		o.Debug = true
	default:
		return o, errors.New("debug must be a boolean")
	}
	return o, nil
}

// createPage accepts multipart form: file=<image>, context=<PageContext JSON, optional>,
// pipeline=<PipelineConfig JSON, optional>, debug=<bool, optional: keep per-stage images>.
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
	opts, err := parseJobOptions(r)
	if err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}

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

	now := time.Now()
	job := jobs.Job{
		JobID:      id,
		PageID:     id,
		InputPath:  inputRel,
		OutputPath: filepath.Join(rel, "output.png"),
		ResultPath: filepath.Join(rel, "result.json"),
		Context:    opts.Context,
		Pipeline:   opts.Pipeline,
		Status:     jobs.StatusQueued,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	if opts.Debug {
		job.DebugDir = filepath.Join(rel, "debug")
	}
	// stored before it is published: a worker that fails fast can report back before Publish
	// returns, and a result for a job the store does not know yet was dropped (queued forever)
	s.store.Put(job)
	if err := s.q.Publish(r.Context(), job); err != nil {
		job.Status, job.Error, job.UpdatedAt = jobs.StatusFailed, "queue unavailable", time.Now()
		s.store.Put(job)
		httpError(w, http.StatusServiceUnavailable, "queue unavailable")
		return
	}
	if cur, ok := s.store.Get(job.JobID); ok && cur.Status != jobs.StatusQueued {
		job = cur // the worker already answered
	}
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

// getJobDebug serves one stage image (detect, mask, inpainted, final) of a debug job.
func (s *server) getJobDebug(w http.ResponseWriter, r *http.Request) {
	stage := r.PathValue("stage")
	if !debugStages[stage] {
		httpError(w, http.StatusNotFound, "unknown stage, want detect|mask|inpainted|final")
		return
	}
	s.serveArtifact(w, r, func(j jobs.Job) string {
		if j.DebugDir == "" {
			return ""
		}
		return filepath.Join(j.DebugDir, stage+".png")
	})
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
	rel := pick(job)
	if rel == "" {
		httpError(w, http.StatusNotFound, "job was not run with debug=true")
		return
	}
	path := filepath.Join(s.cfg.StorageDir, rel)
	if _, err := os.Stat(path); err != nil {
		httpError(w, http.StatusNotFound, "file not found")
		return
	}
	http.ServeFile(w, r, path)
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
