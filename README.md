# Smart Academic Syllabus Assistant

A retrieval-augmented chatbot for searching university engineering syllabus PDFs.
It indexes syllabus text and metadata in Qdrant, retrieves relevant passages for
each question, and uses Google Gemini to produce a grounded answer with PDF page
citations.

## How it works

### 1. Ingest syllabus PDFs

The ingestion pipeline recursively reads PDFs from a directory or ZIP archive.
PyMuPDF extracts text page by page, preserving the original page numbers. Folder
names and syllabus text are used as evidence for metadata such as academic
session, branch, year, course code, subject, and unit. Unknown metadata is left
unset rather than guessed.

Each page is split into overlapping text chunks (800 characters with 150
characters of overlap by default). Sentence Transformers generates normalized
384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2`. The
chunks, vectors, and metadata are uploaded in batches to the existing Qdrant
collection, `pdf_documents`. Unchanged documents are skipped on subsequent
ingestion runs.

### 2. Retrieve evidence

For a question, the application embeds a retrieval query and searches Qdrant for
the nearest syllabus chunks. It can apply metadata constraints inferred from
explicit details in the question, such as a branch or academic session. For
course-specific questions, matching chunks from the same PDF page are gathered
in page order so a complete unit or topic list can be considered together.

### 3. Generate a cited answer

The retrieved PDF excerpts and the current question are sent to the configured
Gemini model. Conversation history may help resolve follow-up references, but it
is not treated as syllabus evidence. Answers are instructed to stay grounded in
the retrieved passages and cite their source markers. Source markers are
formatted with the original PDF filename and page number.

### 4. Use the chat interface

The Gradio interface has a message composer and a **New chat** button; it has no
filter dropdowns. Ask for a subject, course code, branch, session, or unit
directly in your message. If the requested syllabus detail is not in the
retrieved documents, the assistant should say so rather than invent it.

## Example from the running application

The following is a current UI example captured from the running Gradio app.

**Question**

> Give me Unit 1 syllabus topics for Sensors and its Applications, CSE (IoT),
> 2026-27. Cite the source page.

**Answer**

> Here are the Unit 1 syllabus topics for **Sensors and its Applications**
> (Course Code: **CCSIOT0303**) for CSE (IoT), 2026-27:
>
> ### Unit 1: Sensors & Transducers (7 hours)
>
> - **Sensors & Transducers:** Definition, classification of sensors.
> - **Parameter measurements:**
>   - **Displacement:** Potentiometer, LVDT (Linear Variable Differential
>     Transformer), and optical encoder.
>   - **Force:** Strain gauge.
>   - **Pressure:** LVDT-based diaphragm and piezoelectric sensor.
> - **Intelligent sensors:** General structure of smart sensors and its
>   components; characteristics of smart sensors.
> - **Sensors in computers and mobile phones:** Temperature, pressure, humidity,
>   sound, magnetic field, barometer, and gyroscope sensors.
>
> **Source:** *B.Tech in Computer Science and Engineering (Internet of Things)
> (IoT) Second Year 2026-27.pdf*, page 19.

The captured UI also displayed three source labels (`[S3]`, `[S4]`, `[S1]`),
all pointing to that same PDF and page. These labels represented retrieved text
chunks from one page, not three different documents.

## Requirements

- Python
- Google Gemini API key
- Qdrant Cloud URL and API key
- Internet access for the configured APIs and the embedding model

Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in the credentials locally:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.5-flash
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=pdf_documents
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Never commit `.env` or paste API keys into source code. The `.gitignore` excludes
`.env`, the local virtual environment, the source syllabus ZIP, and generated
preprocessing reports.

## Run

Launch the chatbot from the project root:

```powershell
python app.py
```

The Gradio task in `.vscode/tasks.json` can also be used from VS Code. Open the
local URL printed by Gradio (typically `http://127.0.0.1:7860`).

To index PDFs from a ZIP archive or directory, run:

```powershell
python -m scripts.ingest_pdfs --input Syllabus_Repository.zip
```

Use `--input` with a PDF file, ZIP archive, or extracted directory. If omitted,
the command uses `Syllabus_Repository.zip` in the project root. Run
`python -m scripts.ingest_pdfs --help` to see the available options. The
ingestion command requires a compatible existing Qdrant collection; it does not
delete or recreate the collection.

## Tests

Run the unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```
