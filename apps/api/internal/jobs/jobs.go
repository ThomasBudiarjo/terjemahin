package jobs

import (
	"encoding/json"
	"sync"
	"time"
)

type Status string

const (
	StatusQueued Status = "queued"
	StatusDone   Status = "done"
	StatusFailed Status = "failed"
)

// Job mirrors the worker's JobMessage; Context is passed through opaque so the API
// does not need to know the worker's schema.
type Job struct {
	JobID      string          `json:"job_id"`
	PageID     string          `json:"page_id"`
	InputPath  string          `json:"input_path"`
	OutputPath string          `json:"output_path"`
	ResultPath string          `json:"result_path"`
	Context    json.RawMessage `json:"context,omitempty"`
	// Pipeline is the worker's PipelineConfig (method per stage + options), passed through.
	Pipeline json.RawMessage `json:"pipeline,omitempty"`
	// DebugDir, relative to the storage dir, receives per-stage images when set.
	DebugDir string `json:"debug_dir,omitempty"`

	Status    Status    `json:"status"`
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Result is what the worker publishes to the results queue.
type Result struct {
	JobID  string `json:"job_id"`
	PageID string `json:"page_id"`
	Status Status `json:"status"`
	Error  string `json:"error,omitempty"`
}

type Store interface {
	Put(j Job)
	Get(id string) (Job, bool)
	ApplyResult(r Result)
}

type MemoryStore struct {
	mu   sync.RWMutex
	jobs map[string]Job
}

func NewMemoryStore() *MemoryStore { return &MemoryStore{jobs: map[string]Job{}} }

func (s *MemoryStore) Put(j Job) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[j.JobID] = j
}

// IDs lists the stored job ids (tests and diagnostics).
func (s *MemoryStore) IDs() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.jobs))
	for id := range s.jobs {
		out = append(out, id)
	}
	return out
}

func (s *MemoryStore) Get(id string) (Job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	j, ok := s.jobs[id]
	return j, ok
}

func (s *MemoryStore) ApplyResult(r Result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[r.JobID]
	if !ok {
		return
	}
	j.Status = r.Status
	j.Error = r.Error
	j.UpdatedAt = time.Now()
	s.jobs[r.JobID] = j
}
