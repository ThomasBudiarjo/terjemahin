package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thomas/terjemahin/api/internal/config"
	"github.com/thomas/terjemahin/api/internal/jobs"
)

type fakeQueue struct{ published [][]byte }

func (f *fakeQueue) Publish(_ context.Context, j jobs.Job) error {
	b, err := json.Marshal(j)
	f.published = append(f.published, b)
	return err
}

func newTestServer(t *testing.T) (http.Handler, *jobs.MemoryStore, *fakeQueue, string) {
	t.Helper()
	dir := t.TempDir()
	store := jobs.NewMemoryStore()
	q := &fakeQueue{}
	return New(config.Config{StorageDir: dir}, store, q), store, q, dir
}

func pageRequest(t *testing.T, fields map[string]string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("file", "page.png")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = fw.Write([]byte("\x89PNG fake"))
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/v1/pages", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

func post(t *testing.T, h http.Handler, fields map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, pageRequest(t, fields))
	return rec
}

func TestCreatePagePassesPipelineAndDebug(t *testing.T) {
	h, _, q, _ := newTestServer(t)
	pipeline := `{"detector":"ctd","ocr":"manga_ocr","translator":"nllb","mask_refiner":"default",` +
		`"inpainter":"lama_manga","renderer":"default","source_lang":"ja","target_lang":"id",` +
		`"options":{"inpainter":{"size":1024}}}`
	rec := post(t, h, map[string]string{
		"pipeline": pipeline,
		"debug":    "true",
		"context":  `{"series_title":"X"}`,
	})
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if len(q.published) != 1 {
		t.Fatalf("published %d messages", len(q.published))
	}
	t.Logf("published: %s", q.published[0])
	var msg map[string]json.RawMessage
	if err := json.Unmarshal(q.published[0], &msg); err != nil {
		t.Fatal(err)
	}
	var got, want any
	_ = json.Unmarshal(msg["pipeline"], &got)
	_ = json.Unmarshal([]byte(pipeline), &want)
	if !jsonEqual(got, want) {
		t.Errorf("pipeline = %s", msg["pipeline"])
	}
	var debugDir, jobID string
	_ = json.Unmarshal(msg["debug_dir"], &debugDir)
	_ = json.Unmarshal(msg["job_id"], &jobID)
	if debugDir != filepath.Join("pages", jobID, "debug") {
		t.Errorf("debug_dir = %q", debugDir)
	}
	if string(msg["context"]) != `{"series_title":"X"}` {
		t.Errorf("context = %s", msg["context"])
	}
}

func TestCreatePageWithoutOptionalFields(t *testing.T) {
	h, _, q, _ := newTestServer(t)
	if rec := post(t, h, nil); rec.Code != http.StatusAccepted {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	s := string(q.published[0])
	for _, key := range []string{`"pipeline"`, `"debug_dir"`, `"context"`} {
		if strings.Contains(s, key) {
			t.Errorf("%s should be omitted: %s", key, s)
		}
	}
}

func TestCreatePageRejectsBadFields(t *testing.T) {
	cases := map[string]map[string]string{
		"pipeline not json":   {"pipeline": "{nope"},
		"pipeline not object": {"pipeline": `["ctd"]`},
		"unknown field":       {"pipeline": `{"detektor":"ctd"}`},
		"wrong type":          {"pipeline": `{"detector":3}`},
		"options bad stage":   {"pipeline": `{"options":{"bogus":{}}}`},
		"options not object":  {"pipeline": `{"options":{"inpainter":5}}`},
		"trailing data":       {"pipeline": `{} {}`},
		"bad debug":           {"debug": "maybe"},
		"bad context":         {"context": "{"},
	}
	for name, fields := range cases {
		t.Run(name, func(t *testing.T) {
			h, _, q, _ := newTestServer(t)
			rec := post(t, h, fields)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if len(q.published) != 0 {
				t.Error("bad request must not be queued")
			}
		})
	}
}

func TestGetJobDebug(t *testing.T) {
	h, store, _, dir := newTestServer(t)
	rec := post(t, h, map[string]string{"debug": "1"})
	var job jobs.Job
	_ = json.Unmarshal(rec.Body.Bytes(), &job)
	plain := post(t, h, nil)
	var plainJob jobs.Job
	_ = json.Unmarshal(plain.Body.Bytes(), &plainJob)

	get := func(path string) *httptest.ResponseRecorder {
		r := httptest.NewRecorder()
		h.ServeHTTP(r, httptest.NewRequest(http.MethodGet, path, nil))
		return r
	}
	if r := get("/v1/jobs/" + job.JobID + "/debug/mask"); r.Code != http.StatusConflict {
		t.Errorf("queued job: status %d", r.Code)
	}
	store.ApplyResult(jobs.Result{JobID: job.JobID, Status: jobs.StatusDone})
	store.ApplyResult(jobs.Result{JobID: plainJob.JobID, Status: jobs.StatusDone})

	if r := get("/v1/jobs/" + job.JobID + "/debug/mask"); r.Code != http.StatusNotFound {
		t.Errorf("missing file: status %d", r.Code)
	}
	dbg := filepath.Join(dir, job.DebugDir)
	if err := os.MkdirAll(dbg, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dbg, "mask.png"), []byte("png-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r := get("/v1/jobs/" + job.JobID + "/debug/mask"); r.Code != http.StatusOK || r.Body.String() != "png-bytes" {
		t.Errorf("debug image: status %d body %q", r.Code, r.Body)
	}
	if r := get("/v1/jobs/" + job.JobID + "/debug/secret"); r.Code != http.StatusNotFound {
		t.Errorf("unknown stage: status %d", r.Code)
	}
	if r := get("/v1/jobs/" + plainJob.JobID + "/debug/final"); r.Code != http.StatusNotFound {
		t.Errorf("job without debug: status %d", r.Code)
	}
	if r := get("/v1/jobs/nope/debug/final"); r.Code != http.StatusNotFound {
		t.Errorf("unknown job: status %d", r.Code)
	}
}

func jsonEqual(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}

func TestCreatePageRejectsContextTheWorkerCannotRead(t *testing.T) {
	h, _, q, _ := newTestServer(t)
	for _, c := range []string{`123`, `"x"`, `[1]`, `{"glossary":"oops"}`, `{"glossary":[{"source":"a"}]}`,
		`{"previous_pages":[["a"], 2]}`, `{"series_title": 5}`, `{bad`} {
		rec := post(t, h, map[string]string{"context": c})
		if rec.Code != http.StatusBadRequest {
			t.Errorf("context %s: status %d, want 400 (%s)", c, rec.Code, rec.Body)
		}
	}
	if len(q.published) != 0 {
		t.Fatalf("published %d messages for invalid contexts", len(q.published))
	}
	ok := `{"series_title":"X","glossary":[{"source":"a","target":"b"}],"previous_pages":[["x"]]}`
	if rec := post(t, h, map[string]string{"context": ok}); rec.Code != http.StatusAccepted {
		t.Fatalf("valid context: status %d: %s", rec.Code, rec.Body)
	}
}

// syncQueue stands in for a worker that fails before Publish returns.
type syncQueue struct{ store *jobs.MemoryStore }

func (q syncQueue) Publish(_ context.Context, j jobs.Job) error {
	q.store.ApplyResult(jobs.Result{JobID: j.JobID, PageID: j.PageID, Status: jobs.StatusFailed, Error: "boom"})
	return nil
}

func TestFastWorkerResultIsNotLost(t *testing.T) {
	store := jobs.NewMemoryStore()
	h := New(config.Config{StorageDir: t.TempDir()}, store, syncQueue{store})
	rec := post(t, h, map[string]string{})
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var j jobs.Job
	if err := json.Unmarshal(rec.Body.Bytes(), &j); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Get(j.JobID)
	if got.Status != jobs.StatusFailed || got.Error != "boom" {
		t.Fatalf("job status %q (%q), want the worker's failure", got.Status, got.Error)
	}
}

type downQueue struct{}

func (downQueue) Publish(context.Context, jobs.Job) error { return errors.New("down") }

func TestPublishFailureMarksJobFailed(t *testing.T) {
	store := jobs.NewMemoryStore()
	h := New(config.Config{StorageDir: t.TempDir()}, store, downQueue{})
	rec := post(t, h, map[string]string{})
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d", rec.Code)
	}
	if len(store.IDs()) != 1 {
		t.Fatalf("stored %d jobs, want 1", len(store.IDs()))
	}
	for _, id := range store.IDs() {
		if j, _ := store.Get(id); j.Status != jobs.StatusFailed {
			t.Fatalf("job %s left %q", id, j.Status)
		}
	}
}
