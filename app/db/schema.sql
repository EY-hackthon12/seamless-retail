CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    loyalty_tier VARCHAR(50), -- e.g., Gold
    loyalty_points INT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    channel_source VARCHAR(50), -- 'mobile', 'web', 'kiosk'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_summary TEXT -- "User looking for wedding shoes, has blue suit"
);

CREATE TABLE IF NOT EXISTS cart_items (
    item_id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id),
    product_id INT,
    status VARCHAR(50) -- 'added', 'purchased', 'abandoned'
);

CREATE TABLE IF NOT EXISTS conversation_history (
    msg_id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id),
    sender VARCHAR(50), -- 'user', 'agent'
    message_content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
