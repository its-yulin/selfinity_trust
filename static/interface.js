const chatMessagesDiv = document.getElementById("chat-messages");
const userInputElem = document.getElementById("user-input");
const modelToggle = document.getElementById("model-toggle");
const modelLabelLeft = document.getElementById("model-label-left");
const modelLabelRight = document.getElementById("model-label-right");
// State variables
let modelName = modelToggle.checked ? "gpt-4" : "gpt-3.5-turbo";
let messages = [];
let systemMessageRef = null;
let autoScrollState = true;

// Event listener functions
function handleModelToggle() {
  if (modelToggle.checked) {
    modelLabelRight.textContent = "GPT-4";
    modelName = "gpt-4";
  } else {
    modelLabelLeft.textContent = "GPT-3.5";
    modelName = "gpt-3.5-turbo";
  }
}

function handleInputKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("submitBtn").click();
  }
}

function autoScroll() {
  if (autoScrollState) {
    chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
  }
}

// Event listeners for functions above
modelToggle.addEventListener("change", handleModelToggle);
document.getElementById("user-input").addEventListener("keydown", handleInputKeydown);

document;
chatMessagesDiv.addEventListener("scroll", function () {
  const isAtBottom =
    chatMessagesDiv.scrollHeight - chatMessagesDiv.clientHeight <=
    chatMessagesDiv.scrollTop + 1;

  autoScrollState = isAtBottom;
});

window.renderMarkdown = function (content) {
  const md = new markdownit();
  return md.render(content);
};

function highlightCode(element) {
  const codeElements = element.querySelectorAll("pre code");
  codeElements.forEach((codeElement) => {
    hljs.highlightElement(codeElement);
  });
}

function addMessageToDiv(role, content = "") {
    let messageDiv = document.createElement("div");
    messageDiv.className =
        role === "user" ? "message user-message" : "message assistant-message";

    let messageText = document.createElement("p");
    messageText.className = "message-content";
    messageDiv.appendChild(messageText);

    if (content) {
        let renderedContent = window.renderMarkdown(content).trim();
        messageText.innerHTML = renderedContent;
        highlightCode(messageDiv);
    }


    chatMessagesDiv.appendChild(messageDiv);

    autoScroll();

    return messageText;
}


function addTrustAndSourceToDiv(messageText, trust_label, trust_value, source_type, source_value) {
    let container = document.createElement("div");
    container.className = "trust-and-source-container";

    let trustQuestion = document.createElement("div");
    trustQuestion.textContent = "Should we trust this message?";
    trustQuestion.className = "trust-question";
    container.appendChild(trustQuestion);

    let collapsibleContent = document.createElement("div");
    collapsibleContent.className = "collapsible-content";
    collapsibleContent.style.display = "none";

    let trustInfo = document.createElement("div");
    trustInfo.textContent = `${trust_label}. ${trust_value}`;
    collapsibleContent.appendChild(trustInfo);

    let sourceInfo = document.createElement("div");
    sourceInfo.className = "source-info";

    switch (source_type) {
        case 'bookmark':
            let bookmarkLabel = document.createElement("p");
            bookmarkLabel.textContent = "Source: Bookmarks";
            sourceInfo.appendChild(bookmarkLabel);

            let bookmarkList = document.createElement("ul");
            for (const [key, link] of Object.entries(JSON.parse(source_value))) {
                let listItem = document.createElement("li");
                let anchor = document.createElement("a");
                anchor.href = link;
                anchor.textContent = `Link ${key}`;
                listItem.appendChild(anchor);
                bookmarkList.appendChild(listItem);
            }
            sourceInfo.appendChild(bookmarkList);
            break;

        case 'transaction':
            let transactionLabel = document.createElement("p");
            transactionLabel.textContent = "Source: Transaction";
            sourceInfo.appendChild(transactionLabel);

            let transactionLink = document.createElement("a");
            transactionLink.href = source_value;
            transactionLink.textContent = "download relevant transactions";
            transactionLink.download = "";
            sourceInfo.appendChild(transactionLink);
            break;

        case 'gmail':
            let gmailLabel = document.createElement("p");
            gmailLabel.textContent = "Source: Gmail";
            sourceInfo.appendChild(gmailLabel);

            let gmailLink = document.createElement("a");
            gmailLink.href = source_value;
            gmailLink.textContent = "download relevant emails";
            gmailLink.download = "";
            sourceInfo.appendChild(gmailLink);
            break;

        case 'pdf':
            let pdfLabel = document.createElement("p");
            pdfLabel.textContent = "Source: PDFs";
            sourceInfo.appendChild(pdfLabel);

            let pdfList = document.createElement("ul");
            for (const [pdfName, path] of Object.entries(JSON.parse(source_value))) {
                let listItem = document.createElement("li");
                let anchor = document.createElement("a");
                anchor.href = path;
                anchor.textContent = pdfName;
                anchor.download = "";
                listItem.appendChild(anchor);
                pdfList.appendChild(listItem);
            }
            sourceInfo.appendChild(pdfList);
            break;

        default:
            let unsupportedLabel = document.createElement("p");
            unsupportedLabel.textContent = "Unsupported source type";
            sourceInfo.appendChild(unsupportedLabel);
    }

    collapsibleContent.appendChild(sourceInfo);
    container.appendChild(collapsibleContent);

    trustQuestion.addEventListener("click", function() {
        this.classList.toggle("active");
        collapsibleContent.style.display = collapsibleContent.style.display === "none" ? "block" : "none";

        if (trust_label.toLowerCase() === 'yes') {
            collapsibleContent.classList.toggle("trust-yes", this.classList.contains("active"));
            collapsibleContent.classList.remove("trust-no");
        } else if (trust_label.toLowerCase() === 'no') {
            collapsibleContent.classList.toggle("trust-no", this.classList.contains("active"));
            collapsibleContent.classList.remove("trust-yes");
        }
    });

    messageText.appendChild(container);
}
