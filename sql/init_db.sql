CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    location_name VARCHAR(255) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    road_name VARCHAR(255),
    city VARCHAR(100) DEFAULT 'Paris'
);

CREATE TABLE IF NOT EXISTS traffic_data (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    timestamp TIMESTAMP NOT NULL,
    flow_rate DOUBLE PRECISION,
    occupancy_rate DOUBLE PRECISION,
    congestion_level VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    timestamp TIMESTAMP NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pollution_data (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    timestamp TIMESTAMP NOT NULL,
    no2 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    pm25 DOUBLE PRECISION,
    air_quality_index VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    location_id INTEGER REFERENCES locations(id),
    prediction_timestamp TIMESTAMP NOT NULL,
    target_timestamp TIMESTAMP NOT NULL,
    predicted_congestion_level VARCHAR(50),
    predicted_flow_rate DOUBLE PRECISION,
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);