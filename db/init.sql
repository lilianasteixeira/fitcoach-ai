CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT,
    search_strategy TEXT,
    prompt_template TEXT,
    latency_seconds REAL,
    topic TEXT,
    feedback SMALLINT,  -- 1 = positive, -1 = negative, NULL = no feedback
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_feedback ON conversations (feedback);
