-- CREATE TABLE IF NOT EXISTS extracted_text (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     text_content TEXT,
--     image_path TEXT,
--     confidence REAL,
--     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
-- );


-- CREATE TABLE IF NOT EXISTS larvae_measurements (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     tray_number INTEGER,
--     larva_id INTEGER, -- Unique ID for each detected larva within a tray/image
--     length_mm REAL,
--     width_mm REAL,
--     area_sq_mm REAL,
--     estimated_weight_mg REAL,
--     confidence REAL,
--     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
--     FOREIGN KEY (tray_number) REFERENCES extracted_text(text_content) -- Link to tray number
-- );


-- SELECT * FROM extracted_text;
SELECT * FROM larvae_measurements;