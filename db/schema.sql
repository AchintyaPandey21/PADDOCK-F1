-- Paddock F1 PostgreSQL Relational Schema
-- Normalized relational schema for Formula 1 race sessions and analytics

-- 1. Seasons
CREATE TABLE IF NOT EXISTS seasons (
    year INT PRIMARY KEY
);

-- 2. Circuits
CREATE TABLE IF NOT EXISTS circuits (
    circuit_id SERIAL PRIMARY KEY,
    circuit_key VARCHAR(50) UNIQUE,
    circuit_name VARCHAR(150) NOT NULL,
    location VARCHAR(100),
    country VARCHAR(100)
);

-- 3. Teams
CREATE TABLE IF NOT EXISTS teams (
    team_id SERIAL PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL UNIQUE,
    team_color VARCHAR(10)
);

-- 4. Drivers
CREATE TABLE IF NOT EXISTS drivers (
    driver_id SERIAL PRIMARY KEY,
    abbreviation VARCHAR(10) NOT NULL UNIQUE,
    driver_number INT,
    full_name VARCHAR(150) NOT NULL,
    broadcast_name VARCHAR(100),
    country_code VARCHAR(10)
);

-- 5. Races (Grand Prix Events)
CREATE TABLE IF NOT EXISTS races (
    race_id SERIAL PRIMARY KEY,
    season_year INT NOT NULL REFERENCES seasons(year) ON DELETE CASCADE,
    round_number INT NOT NULL,
    event_name VARCHAR(150) NOT NULL,
    official_event_name VARCHAR(255),
    event_date DATE,
    event_format VARCHAR(50),
    circuit_id INT REFERENCES circuits(circuit_id),
    CONSTRAINT uq_race_season_round UNIQUE (season_year, round_number),
    CONSTRAINT uq_race_season_event UNIQUE (season_year, event_name)
);

-- 6. Sessions (e.g. Race, Qualifying, Practice 1, Sprint)
CREATE TABLE IF NOT EXISTS sessions (
    session_id SERIAL PRIMARY KEY,
    race_id INT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
    session_name VARCHAR(50) NOT NULL,
    session_type VARCHAR(50),
    session_date TIMESTAMPTZ,
    is_loaded BOOLEAN DEFAULT FALSE,
    CONSTRAINT uq_session_race_name UNIQUE (race_id, session_name)
);

-- 7. Race Results (and Qualifying Results)
CREATE TABLE IF NOT EXISTS race_results (
    result_id SERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    driver_id INT NOT NULL REFERENCES drivers(driver_id),
    team_id INT REFERENCES teams(team_id),
    position INT,
    classified_position VARCHAR(10),
    grid_position INT,
    time_seconds DOUBLE PRECISION,
    status VARCHAR(100),
    points DOUBLE PRECISION,
    laps_completed INT,
    q1_seconds DOUBLE PRECISION,
    q2_seconds DOUBLE PRECISION,
    q3_seconds DOUBLE PRECISION,
    CONSTRAINT uq_result_session_driver UNIQUE (session_id, driver_id)
);

-- 8. Laps
CREATE TABLE IF NOT EXISTS laps (
    lap_id BIGSERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    driver_id INT NOT NULL REFERENCES drivers(driver_id),
    team_id INT REFERENCES teams(team_id),
    lap_number INT NOT NULL,
    position INT,
    lap_time_seconds DOUBLE PRECISION,
    sector1_time_seconds DOUBLE PRECISION,
    sector2_time_seconds DOUBLE PRECISION,
    sector3_time_seconds DOUBLE PRECISION,
    compound VARCHAR(20),
    stint INT,
    tyre_life INT,
    fresh_tyre BOOLEAN,
    track_status VARCHAR(20),
    is_accurate BOOLEAN,
    is_timed_lap BOOLEAN,
    is_push_lap BOOLEAN,
    pit_in_time_seconds DOUBLE PRECISION,
    pit_out_time_seconds DOUBLE PRECISION,
    CONSTRAINT uq_lap_session_driver_number UNIQUE (session_id, driver_id, lap_number)
);

-- 9. Tyre Stints
CREATE TABLE IF NOT EXISTS tyre_stints (
    stint_id SERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    driver_id INT NOT NULL REFERENCES drivers(driver_id),
    team_id INT REFERENCES teams(team_id),
    stint_number INT NOT NULL,
    compound VARCHAR(20) NOT NULL,
    start_lap INT NOT NULL,
    end_lap INT NOT NULL,
    stint_length INT NOT NULL,
    mean_lap_seconds DOUBLE PRECISION,
    fresh_tyre BOOLEAN,
    mean_tyre_life DOUBLE PRECISION,
    CONSTRAINT uq_stint_session_driver_number UNIQUE (session_id, driver_id, stint_number)
);

-- 10. Pit Stops
CREATE TABLE IF NOT EXISTS pit_stops (
    pit_stop_id SERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    driver_id INT NOT NULL REFERENCES drivers(driver_id),
    lap_number INT NOT NULL,
    pit_in_time_seconds DOUBLE PRECISION,
    pit_out_time_seconds DOUBLE PRECISION,
    duration_seconds DOUBLE PRECISION,
    CONSTRAINT uq_pit_session_driver_lap UNIQUE (session_id, driver_id, lap_number)
);

-- 11. Weather Observations
CREATE TABLE IF NOT EXISTS weather_observations (
    weather_id SERIAL PRIMARY KEY,
    session_id INT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    sample_time_seconds DOUBLE PRECISION,
    air_temp DOUBLE PRECISION,
    track_temp DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction INT,
    rainfall BOOLEAN
);

-- Indexes for optimal analytics query performance
CREATE INDEX IF NOT EXISTS idx_races_season ON races(season_year);
CREATE INDEX IF NOT EXISTS idx_sessions_race ON sessions(race_id);
CREATE INDEX IF NOT EXISTS idx_results_session ON race_results(session_id);
CREATE INDEX IF NOT EXISTS idx_laps_session_driver ON laps(session_id, driver_id);
CREATE INDEX IF NOT EXISTS idx_laps_session_lap ON laps(session_id, lap_number);
CREATE INDEX IF NOT EXISTS idx_stints_session_driver ON tyre_stints(session_id, driver_id);
CREATE INDEX IF NOT EXISTS idx_weather_session ON weather_observations(session_id);
