const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function ask(question, signal = null) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
