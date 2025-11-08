import { useState } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export default function App() {
  const [message, setMessage] = useState('Hi, I am looking for shoes for a wedding.')
  const [trace, setTrace] = useState<string[]>([])
  const [route, setRoute] = useState('')

  const send = async () => {
    const res = await axios.post(`${API_BASE}/api/v1/chat`, { message })
    setTrace(res.data.trace)
    setRoute(res.data.route)
  }

  return (
    <div className="max-w-2xl mx-auto p-8 space-y-4">
      <h1 className="text-2xl font-bold">Seamless Retail Demo</h1>
      <div className="flex gap-2">
        <input className="flex-1 border rounded p-2" value={message} onChange={e=>setMessage(e.target.value)} />
        <button className="bg-black text-white px-4 py-2 rounded" onClick={send}>Send</button>
      </div>
      <div className="border rounded p-4 bg-white">
        <div className="text-sm text-gray-500">Route: {route}</div>
        <ul className="list-disc pl-6">
          {trace.map((t, i) => <li key={i}>{t}</li>)}
        </ul>
      </div>
    </div>
  )
}
