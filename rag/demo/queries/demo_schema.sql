CREATE TABLE IF NOT EXISTS runs(
  id SERIAL CONSTRAINT PK_RUN_ID PRIMARY KEY,
  start_datetime TIMESTAMPTZ DEFAULT now(),
  end_datetime TIMESTAMPTZ,
  doc_viewed_count INT,
  doc_processed_count INT,
  doc_failed_count INT,
  doc_ignored_count INT,
  parameters JSONB
);

CREATE TABLE IF NOT EXISTS documents(
  id SERIAL CONSTRAINT PK_DOCUMENT_ID PRIMARY KEY,
  run_id INT CONSTRAINT FK_DOCUMENT_RUN REFERENCES runs(id),
  filename TEXT,
  file_extension TEXT,
  source_folder_path TEXT,
  export_folder_path TEXT,
  file_hash TEXT,
  start_datetime TIMESTAMPTZ DEFAULT now(),
  end_datetime TIMESTAMPTZ,
  process_status TEXT,
  process_time INT,
  error_type TEXT,
  error_message TEXT,
  page_count INT,
  text_count INT,
  table_count INT,
  image_count INT,
  departement TEXT -- Used for Row-Level Security (RLS)
);
