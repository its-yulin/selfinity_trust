function updateSystemMessage(systemMessage) {
  if (
    systemMessage &&
    (!systemMessageRef || systemMessage !== systemMessageRef.content)
  ) {
    let systemMessageIndex = messages.findIndex((message) => message.role === "system");
    // If the system message exists in array, remove it
    if (systemMessageIndex !== -1) {
      messages.splice(systemMessageIndex, 1);
    }
    systemMessageRef = { role: "system", content: systemMessage };
    messages.push(systemMessageRef);
  }
}

async function postRequest() {
  return await fetch("/gpt4", {
    method: "POST",
    body: JSON.stringify({
      messages: messages,
      model_type: modelName,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  });
}
async function handleResponse(response, messageText) {
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Assuming the entire response is JSON and not a stream
    const responseData = await response.json();

    // Extract values from the JSON response
    const { response: assistantMessage, source, trust: trustString } = responseData;
    const { source_type, source_value } = source;
    const trust = JSON.parse(trustString);
    const { trust_type, explanation: trust_value } = trust;

    // Update the UI with the assistant message
    messageText.innerHTML = window.renderMarkdown(assistantMessage).trim();
    highlightCode(messageText);
    addTrustAndSourceToDiv(messageText, trust_type, trust_value, source_type, source_value);
    autoScroll();

    // Push to messages array for any further use
    messages.push({
        role: "assistant",
        content: assistantMessage,
    });
}


window.onload = function () {
  document.getElementById("chat-form").addEventListener("submit", async function (event) {
    event.preventDefault();

    let userInput = userInputElem.value.trim();
    let systemMessage = document.getElementById("system-message").value.trim();

    updateSystemMessage(systemMessage);

    messages.push({ role: "user", content: userInput });
    addMessageToDiv("user", userInput);

    userInputElem.value = "";

    let messageText = addMessageToDiv("assistant");

    const response = await postRequest();

    handleResponse(response, messageText);
  });
};
