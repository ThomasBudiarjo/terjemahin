package config

import "os"

type Config struct {
	Addr       string
	AMQPURL    string
	AMQPQueue  string
	StorageDir string
}

func Load() Config {
	return Config{
		Addr:       env("API_ADDR", ":8080"),
		AMQPURL:    env("AMQP_URL", "amqp://guest:guest@localhost:5672/"),
		AMQPQueue:  env("AMQP_QUEUE", "terjemahin.jobs"),
		StorageDir: env("STORAGE_DIR", "./data"),
	}
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
