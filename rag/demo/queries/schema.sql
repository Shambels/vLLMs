CREATE TABLE IF NOT EXISTS run(
    run_id SERIAL CONSTRAINT PK_RUN_ID PRIMARY KEY,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    doc_processed INT,
    doc_ignored INT,
    doc_failed INT,
    parameters JSONB
);

CREATE TABLE IF NOT EXISTS document (
    document_id SERIAL CONSTRAINT PK_DOCUMENT_ID PRIMARY KEY,
    run_id INT CONSTRAINT FK_DOCUMENT_RUN REFERENCES run(run_id),
    file_path VARCHAR(255),
    export_folder VARCHAR(255),
    file_name VARCHAR(255),
    file_format_ VARCHAR(255),
    file_hash VARCHAR(255),
    status_ VARCHAR(255),
    page_count INT,
    text_count INT,
    table_count INT,
    image_count INT,
    process_time FLOAT,
    error VARCHAR(255)
)