import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [context, setContext] = useState('')
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [isIndexing, setIsIndexing] = useState(false)

  async function handleAskQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setErrorMessage('')
    setStatusMessage('')

    const trimmedQuestion = question.trim()
    if (!trimmedQuestion) {
      setErrorMessage('Please enter a question first.')
      return
    }

    setIsAsking(true)
    setAnswer('')
    setContext('')

    try {
      const response = await fetch('/qa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: trimmedQuestion }),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.detail ?? 'Unable to fetch answer.')
      }

      setAnswer(data.answer ?? '')
      setContext(data.context ?? '')
      setStatusMessage('Answer generated successfully.')
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Something went wrong.'
      )
    } finally {
      setIsAsking(false)
    }
  }

  async function handlePdfUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setErrorMessage('')
    setStatusMessage('')

    if (!pdfFile) {
      setErrorMessage('Please choose a PDF file to upload.')
      return
    }

    const formData = new FormData()
    formData.append('file', pdfFile)

    setIsIndexing(true)

    try {
      const response = await fetch('/index-pdf', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data?.detail ?? 'Unable to index PDF.')
      }

      const chunks = data?.chunks_indexed
      setStatusMessage(
        typeof chunks === 'number'
          ? `Indexed PDF successfully (${chunks} chunks).`
          : 'Indexed PDF successfully.'
      )
      setPdfFile(null)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Something went wrong.'
      )
    } finally {
      setIsIndexing(false)
    }
  }

  return (
    <main className="app-shell">
      <header>
        <h1>IKMS Multi-Agent RAG</h1>
        <p>Upload your PDF, then ask a question from the indexed knowledge base.</p>
      </header>

      <section className="card">
        <h2>1) Index a PDF</h2>
        <form onSubmit={handlePdfUpload} className="stack">
          <input
            type="file"
            accept="application/pdf"
            onChange={(event) => setPdfFile(event.target.files?.[0] ?? null)}
          />
          <button type="submit" disabled={isIndexing}>
            {isIndexing ? 'Indexing...' : 'Upload and Index'}
          </button>
        </form>
      </section>

      <section className="card">
        <h2>2) Ask a question</h2>
        <form onSubmit={handleAskQuestion} className="stack">
          <textarea
            placeholder="Ask your question here..."
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={4}
          />
          <button type="submit" disabled={isAsking}>
            {isAsking ? 'Thinking...' : 'Ask'}
          </button>
        </form>
      </section>

      {statusMessage ? <p className="status ok">{statusMessage}</p> : null}
      {errorMessage ? <p className="status error">{errorMessage}</p> : null}

      <section className="card">
        <h2>Answer</h2>
        <pre>{answer || 'No answer yet.'}</pre>
      </section>

      <section className="card">
        <h2>Context</h2>
        <pre>{context || 'No context yet.'}</pre>
      </section>
    </main>
  )
}

export default App
