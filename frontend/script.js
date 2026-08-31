(function() {
  const apiUrlInput = document.getElementById("api-url");
  const healthButton = document.getElementById("health-button");
  const healthDot = document.getElementById("health-dot");
  const questionInput = document.getElementById("question-input");
  const askButton = document.getElementById("ask-button");
  const messagesContainer = document.getElementById("messages-container");
  const emptyState = document.getElementById("empty-state");
  const errorBar = document.getElementById("error-bar");

  function getApiUrl() {
    return apiUrlInput.value.trim().replace(/\/$/, "");
  }

  function showError(msg) {
    errorBar.textContent = msg;
    errorBar.classList.add("show");
  }

  function clearError() {
    errorBar.classList.remove("show");
  }

  function addMessage(role, content, sources = null, route = null) {
    // remove empty state if present
    if (emptyState.style.display !== 'none') {
      emptyState.style.display = 'none';
    }

    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;
    wrapper.appendChild(bubble);

    const meta = document.createElement('div');
    meta.className = 'meta';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    meta.innerHTML = `<span>${time}</span>`;
    if (route) meta.innerHTML += `<span>· ${route}</span>`;
    wrapper.appendChild(meta);

    if (sources && sources.length) {
      const srcDiv = document.createElement('div');
      srcDiv.className = 'sources-inline';
      srcDiv.innerHTML = sources.map(s => {
        // API returns `book_name`, `page_start`, `page_end` (or null).
        const book = s.book_name || s.book || 'Unknown Book';
        const ps = s.page_start;
        const pe = s.page_end;

        if (ps !== null && ps !== undefined) {
          if (pe === null || pe === undefined || ps === pe) {
            return `<span>${book} — page ${ps}</span>`;
          }
          return `<span>${book} — pages ${ps}-${pe}</span>`;
        }

        return `<span>${book} — (no page info)</span>`;
      }).join(' ');
      wrapper.appendChild(srcDiv);
    }

    messagesContainer.appendChild(wrapper);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  async function typeAnswer(text, sources, route) {
    // create assistant message with empty bubble
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = '';
    wrapper.appendChild(bubble);

    const meta = document.createElement('div');
    meta.className = 'meta';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    meta.innerHTML = `<span>${time}</span>`;
    if (route) meta.innerHTML += `<span>· ${route}</span>`;
    wrapper.appendChild(meta);

    if (sources && sources.length) {
      const srcDiv = document.createElement('div');
      srcDiv.className = 'sources-inline';
      srcDiv.innerHTML = sources.map(s => {
        const book = s.book_name || s.book || 'Unknown Book';
        const ps = s.page_start;
        const pe = s.page_end;

        if (ps !== null && ps !== undefined) {
          if (pe === null || pe === undefined || ps === pe) {
            return `<span>${book} — page ${ps}</span>`;
          }
          return `<span>${book} — pages ${ps}-${pe}</span>`;
        }

        return `<span>${book} — (no page info)</span>`;
      }).join(' ');
      wrapper.appendChild(srcDiv);
    }

    messagesContainer.appendChild(wrapper);
    
    // type effect
    for (const letter of text) {
      bubble.textContent += letter;
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      await new Promise(resolve => setTimeout(resolve, 8));
    }
  }

  // health check
  healthButton.addEventListener("click", async () => {
    clearError();
    healthButton.textContent = "⏳";
    healthButton.disabled = true;
    healthDot.className = "dot idle";

    try {
      const res = await fetch(`${getApiUrl()}/health`);
      const data = await res.json();
      healthDot.className = data.status === "ok" ? "dot ok" : "dot error";
      if (data.status !== "ok") showError("API health check returned not OK.");
    } catch {
      healthDot.className = "dot error";
      showError("Cannot reach API. Check URL and make sure FastAPI is running.");
    }
    healthButton.textContent = "Ping API";
    healthButton.disabled = false;
  });

  // ask question
  async function askQuestion() {
    clearError();
    const query = questionInput.value.trim();
    if (!query) {
      showError("Please write a question.");
      return;
    }

    // add user message
    addMessage('user', query);
    questionInput.value = '';
    questionInput.style.height = 'auto';

    askButton.disabled = true;
    askButton.textContent = "⏳";

    try {
      const response = await fetch(`${getApiUrl()}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 3 }),
      });
      if (!response.ok) throw new Error(`Server error (${response.status})`);

      const data = await response.json();
      await typeAnswer(data.answer, data.sources, data.route);
    } catch (error) {
      showError(error.message || "Could not get an answer from the API.");
      // add a fallback error message in chat
      addMessage('assistant', '⚠️ The library is currently unreachable. Please check the connection.');
    }

    askButton.disabled = false;
    askButton.textContent = "📜 Send";
  }

  // event listeners
  askButton.addEventListener("click", askQuestion);

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  });

  // auto-resize textarea
  questionInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 130) + 'px';
  });

  // chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.textContent;
      questionInput.focus();
      questionInput.dispatchEvent(new Event('input'));
    });
  });

  // init
  healthDot.className = "dot idle";
})();