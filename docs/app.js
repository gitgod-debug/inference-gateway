// ── Configuration ─────────────────────────────────────────
// REPLACE THIS with your real Render URL after deployment
const API_BASE = "https://inference-gateway.onrender.com";

const ENDPOINTS = {
  health: { method: "GET", path: "/health", label: "Health Check" },
  models: { method: "GET", path: "/v1/models", label: "List Models" },
  chat: {
    method: "POST", path: "/v1/chat/completions", label: "Chat Completion",
    body: {
      model: "llama-3.3-70b-versatile",
      messages: [{ role: "user", content: "Explain what an inference gateway is in 2 sentences." }]
    }
  }
};

// ── State ─────────────────────────────────────────────────
let activeTab = "health";
let isLoading = false;

// ── DOM Ready ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  document.getElementById("run-btn").addEventListener("click", runQuery);
});

// ── Tab Switching ─────────────────────────────────────────
function initTabs() {
  document.querySelectorAll(".playground__tab").forEach(tab => {
    tab.addEventListener("click", () => {
      if (isLoading) return;
      document.querySelectorAll(".playground__tab").forEach(t => t.classList.remove("playground__tab--active"));
      tab.classList.add("playground__tab--active");
      activeTab = tab.dataset.tab;
      const output = document.getElementById("output");
      output.textContent = `Press "Run" to call ${ENDPOINTS[activeTab].label}`;
      output.style.color = "var(--text-muted)";
    });
  });
}

// ── API Call ──────────────────────────────────────────────
async function runQuery() {
  if (isLoading) return;
  const btn = document.getElementById("run-btn");
  const output = document.getElementById("output");
  const endpoint = ENDPOINTS[activeTab];

  isLoading = true;
  btn.disabled = true;
  btn.textContent = "Running...";
  output.style.color = "var(--accent-orange)";
  output.innerHTML = '<span class="spinner"></span> Connecting to gateway...';

  // Cold-start handling
  const wakeTimer = setTimeout(() => {
    output.innerHTML = '<span class="spinner"></span> <span class="wake-msg">Waking up the server — free tier cold start, usually takes 30-50 seconds. Hang tight!</span>';
  }, 5000);

  try {
    const fetchOpts = { method: endpoint.method, headers: { "Content-Type": "application/json" } };
    if (endpoint.body) fetchOpts.body = JSON.stringify(endpoint.body);

    const response = await fetch(API_BASE + endpoint.path, fetchOpts);
    clearTimeout(wakeTimer);

    const data = await response.json();
    output.style.color = "var(--accent-green)";
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    clearTimeout(wakeTimer);
    output.style.color = "var(--accent-red)";

    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      output.innerHTML = `<span style="color: var(--accent-orange)">⚠ Server is not reachable.</span>\n\nThis could mean:\n• The Render deployment URL hasn't been configured yet\n• The server is still waking up (try again in ~30 seconds)\n\nYou can always run this locally:\n  git clone → docker compose up -d → curl localhost:8080/health`;
    } else {
      output.textContent = `Error: ${err.message}`;
    }
  } finally {
    isLoading = false;
    btn.disabled = false;
    btn.textContent = "▶ Run";
  }
}

// ── Smooth scroll for nav links ──────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute("href"));
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});
