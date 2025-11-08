-- Enable extensions used in this schema
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS customers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE,
  name TEXT,
  loyalty_tier TEXT,
  points INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID REFERENCES customers(id),
  channel TEXT,
  started_at TIMESTAMP DEFAULT NOW(),
  last_event_at TIMESTAMP DEFAULT NOW(),
  context JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS carts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID REFERENCES customers(id),
  items JSONB DEFAULT '[]'::jsonb,
  status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID REFERENCES sessions(id),
  type TEXT,
  payload JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
