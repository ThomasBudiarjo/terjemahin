package queue

import (
	"context"
	"encoding/json"
	"log"

	amqp "github.com/rabbitmq/amqp091-go"

	"github.com/thomas/terjemahin/api/internal/jobs"
)

const resultsSuffix = ".results"

type Queue struct {
	conn  *amqp.Connection
	ch    *amqp.Channel
	queue string
}

func Dial(url, queue string) (*Queue, error) {
	conn, err := amqp.Dial(url)
	if err != nil {
		return nil, err
	}
	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, err
	}
	for _, name := range []string{queue, queue + resultsSuffix} {
		if _, err := ch.QueueDeclare(name, true, false, false, false, nil); err != nil {
			conn.Close()
			return nil, err
		}
	}
	return &Queue{conn: conn, ch: ch, queue: queue}, nil
}

func (q *Queue) Close() {
	_ = q.ch.Close()
	_ = q.conn.Close()
}

func (q *Queue) Publish(ctx context.Context, j jobs.Job) error {
	body, err := json.Marshal(j)
	if err != nil {
		return err
	}
	return q.ch.PublishWithContext(ctx, "", q.queue, false, false, amqp.Publishing{
		ContentType:  "application/json",
		DeliveryMode: amqp.Persistent,
		Body:         body,
	})
}

// ConsumeResults applies worker results until ctx is cancelled.
func (q *Queue) ConsumeResults(ctx context.Context, apply func(jobs.Result)) {
	msgs, err := q.ch.Consume(q.queue+resultsSuffix, "", false, false, false, false, nil)
	if err != nil {
		log.Printf("consume results: %v", err)
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case m, ok := <-msgs:
			if !ok {
				return
			}
			var r jobs.Result
			if err := json.Unmarshal(m.Body, &r); err != nil {
				log.Printf("bad result: %v", err)
				_ = m.Nack(false, false)
				continue
			}
			apply(r)
			_ = m.Ack(false)
		}
	}
}
