package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thomas/terjemahin/api/internal/config"
	"github.com/thomas/terjemahin/api/internal/httpapi"
	"github.com/thomas/terjemahin/api/internal/jobs"
	"github.com/thomas/terjemahin/api/internal/queue"
)

func main() {
	cfg := config.Load()

	q, err := queue.Dial(cfg.AMQPURL, cfg.AMQPQueue)
	if err != nil {
		log.Fatalf("amqp: %v", err)
	}
	defer q.Close()

	store := jobs.NewMemoryStore() // TODO: Postgres-backed store
	go q.ConsumeResults(context.Background(), store.ApplyResult)

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           httpapi.New(cfg, store, q),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("api listening on %s", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("http: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}
