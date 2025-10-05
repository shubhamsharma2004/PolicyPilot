export async function ask(question, signal = null) {
  const res = await fetch("http://127.0.0.1:8000/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ question }),
    signal
  });
  return res.json();
}