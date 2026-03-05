// Loading Overlay Management
function showLoading(message = "Analyzing claim data...") {
    const overlay = document.getElementById('loadingOverlay');
    const loaderText = overlay.querySelector('.loader-text');
    
    if (loaderText) {
        loaderText.textContent = message;
    }
    
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('active');
}

// Form Submission Handlers
document.addEventListener('DOMContentLoaded', function() {
    // Session Timer
    let sessionSeconds = 0;
    const sessionTimeEl = document.getElementById('sessionTime');
    
    if (sessionTimeEl) {
        setInterval(() => {
            sessionSeconds++;
            const hours = Math.floor(sessionSeconds / 3600);
            const minutes = Math.floor((sessionSeconds % 3600) / 60);
            const secs = sessionSeconds % 60;
            
            const timeStr = hours > 0 
                ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
                : `${minutes}:${String(secs).padStart(2, '0')}`;
            
            sessionTimeEl.textContent = `Session: ${timeStr}`;
        }, 1000);
    }

    // Smooth scroll on page load
    setTimeout(() => {
        const resultContainer = document.querySelector('.result-container');
        if (resultContainer) {
            resultContainer.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }, 300);
    
    // Fraud Form Handler
    const fraudForm = document.getElementById('fraudForm');
    if (fraudForm) {
        fraudForm.addEventListener('submit', function(e) {
            showLoading("Analyzing claim for fraud patterns...");
        });
    }
    
    // Damage Form Handler
    const damageForm = document.getElementById('damageForm');
    if (damageForm) {
        damageForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('fileUpload');

            // Check if an image is selected
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                showUploadError('Please select an image before submitting.');
                return false;
            }

            showLoading("Assessing vehicle damage...");
        });
    }
    
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Animate elements on scroll
    initScrollAnimations();
    
    // Add input animation effects
    enhanceFormInputs();
    
    // Set dynamic year in footer
    const yearElement = document.getElementById('currentYear');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
});

// Image Upload & Preview
function showUploadError(message) {
    const errorDiv = document.getElementById('uploadError');
    if (errorDiv) {
        errorDiv.textContent = '⚠ ' + message;
        errorDiv.style.display = 'block';
        setTimeout(() => { errorDiv.style.display = 'none'; }, 5000);
    }
}

function previewImage(event) {
    const file = event.target.files[0];

    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showUploadError('Please upload a valid image file (JPG, PNG, or WebP).');
        event.target.value = '';
        return;
    }

    // Validate file size (5MB max)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
        showUploadError('File size must be less than 5MB.');
        event.target.value = '';
        return;
    }

    const reader = new FileReader();

    reader.onload = function(e) {
        const preview = document.getElementById('preview');
        const previewContainer = document.getElementById('previewContainer');
        const uploadArea = document.getElementById('uploadArea');

        preview.src = e.target.result;

        // Show preview with animation
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';

        // Trigger fade-in animation
        setTimeout(() => {
            previewContainer.style.opacity = '1';
        }, 10);
    };

    reader.onerror = function() {
        showUploadError('Error reading file. Please try again.');
        event.target.value = '';
    };

    reader.readAsDataURL(file);
}

function resetUpload() {
    const fileInput = document.getElementById('fileUpload');
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
}

// Drag & Drop Upload Enhancement
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileUpload');
    
    if (uploadArea && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle dropped files
        uploadArea.addEventListener('drop', handleDrop, false);
    }
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    const uploadLabel = document.querySelector('.upload-label');
    if (uploadLabel) {
        uploadLabel.style.borderColor = 'var(--primary-500)';
        uploadLabel.style.background = 'var(--info-bg)';
    }
}

function unhighlight(e) {
    const uploadLabel = document.querySelector('.upload-label');
    if (uploadLabel) {
        uploadLabel.style.borderColor = '';
        uploadLabel.style.background = '';
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    const fileInput = document.getElementById('fileUpload');
    
    if (files.length > 0) {
        fileInput.files = files;
        previewImage({ target: fileInput });
    }
}

// Chatbot Functionality
function toggleChat() {
    const chatWindow = document.getElementById('chatWindow');
    
    if (chatWindow.classList.contains('active')) {
        chatWindow.classList.remove('active');
    } else {
        chatWindow.classList.add('active');
        // Focus on input when opening
        setTimeout(() => {
            const input = document.getElementById('userInput');
            if (input) input.focus();
        }, 300);
    }
}

function handleEnter(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    const inputField = document.getElementById('userInput');
    const message = inputField.value.trim();
    
    if (message === "") return;
    
    // Add user message to chat
    addMessage(message, 'user');
    inputField.value = "";
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    // Send to server
    fetch('/ask_bot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: message })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add bot response
        setTimeout(() => {
            addMessage(data.answer || "I'm sorry, I couldn't process that request.", 'bot');
        }, 300);
    })
    .catch(error => {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        
        setTimeout(() => {
            addMessage("I'm having trouble connecting right now. Please try again in a moment.", 'bot');
        }, 300);
    });
}

function addMessage(text, type) {
    const chatBody = document.getElementById('chatBody');
    
    const messageContainer = document.createElement('div');
    messageContainer.className = `message-container ${type}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<span class="material-icons-round">' + 
                      (type === 'bot' ? 'smart_toy' : 'person') + 
                      '</span>';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `<p>${escapeHtml(text)}</p>`;
    
    messageContainer.appendChild(avatar);
    messageContainer.appendChild(bubble);
    
    chatBody.appendChild(messageContainer);
    scrollChatToBottom();
}

function addTypingIndicator() {
    const chatBody = document.getElementById('chatBody');
    
    const typingContainer = document.createElement('div');
    typingContainer.className = 'message-container bot-message typing-indicator';
    typingContainer.id = 'typing-' + Date.now();
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<span class="material-icons-round">smart_toy</span>';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    
    typingContainer.appendChild(avatar);
    typingContainer.appendChild(bubble);
    
    chatBody.appendChild(typingContainer);
    scrollChatToBottom();
    
    return typingContainer.id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function scrollChatToBottom() {
    const chatBody = document.getElementById('chatBody');
    chatBody.scrollTo({
        top: chatBody.scrollHeight,
        behavior: 'smooth'
    });
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Add typing dots animation CSS dynamically
const typingStyle = document.createElement('style');
typingStyle.textContent = `
    .typing-dots {
        display: flex;
        gap: 4px;
        padding: 8px 0;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--gray-400);
        animation: typing 1.4s ease-in-out infinite;
    }
    
    .typing-dots span:nth-child(1) {
        animation-delay: 0s;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            opacity: 0.3;
            transform: translateY(0);
        }
        30% {
            opacity: 1;
            transform: translateY(-8px);
        }
    }
`;
document.head.appendChild(typingStyle);

// Scroll Animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
}

// Form Input Enhancements

function enhanceFormInputs() {
    const inputs = document.querySelectorAll('.form-input, .form-select');
    
    inputs.forEach(input => {
        // Add focus effects
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
        
        // Add filled state
        input.addEventListener('input', function() {
            if (this.value) {
                this.parentElement.classList.add('filled');
            } else {
                this.parentElement.classList.remove('filled');
            }
        });
    });
}

// Utility Functions
// Format currency inputs
document.addEventListener('DOMContentLoaded', function() {
    const currencyInputs = document.querySelectorAll('input[name*="premium"], input[name*="amount"], input[name*="deductable"], input[name*="limit"], input[name*="gains"], input[name*="loss"]');
    
    currencyInputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.value) {
                const value = parseFloat(this.value);
                if (!isNaN(value)) {
                    this.value = value.toFixed(2);
                }
            }
        });
    });
});

// Auto-hide alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const results = document.querySelectorAll('.result-container');
    
    results.forEach(result => {
        setTimeout(() => {
            result.style.transition = 'opacity 0.5s ease';
            // Don't auto-hide, let user dismiss manually
        }, 5000);
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to open chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        toggleChat();
    }
    
    // Escape to close chat
    if (e.key === 'Escape') {
        const chatWindow = document.getElementById('chatWindow');
        if (chatWindow && chatWindow.classList.contains('active')) {
            toggleChat();
        }
    }
});

// Performance Optimization

// Lazy load images
document.addEventListener('DOMContentLoaded', function() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        observer.unobserve(img);
                    }
                }
            });
        });
        
        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }
});

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Accessibility Enhancements
document.addEventListener('DOMContentLoaded', function() {
    let isUsingMouse = false;
    
    document.addEventListener('mousedown', () => {
        isUsingMouse = true;
    });
    
    document.addEventListener('keydown', () => {
        isUsingMouse = false;
    });
    
    document.addEventListener('focusin', (e) => {
        if (!isUsingMouse) {
            e.target.classList.add('keyboard-focus');
        }
    });
    
    document.addEventListener('focusout', (e) => {
        e.target.classList.remove('keyboard-focus');
    });
});

// Announce messages to screen readers
function announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

// Add screen reader only class
const srOnlyStyle = document.createElement('style');
srOnlyStyle.textContent = `
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border-width: 0;
    }
    
    .keyboard-focus {
        outline: 3px solid var(--primary-500);
        outline-offset: 2px;
    }
`;
document.head.appendChild(srOnlyStyle);

console.log('%c✨ InsureTech 360 ✨', 'color: #1a73e8; font-size: 20px; font-weight: bold;');

// Assistant Card Functions
function sendAssistantMessage() {
    const input = document.getElementById('assistantInput');
    const chatArea = document.getElementById('assistantChatArea');
    
    if (!input || !chatArea) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'assistant-message user';
    userMsg.innerHTML = `
        <div class="assistant-avatar">
            <span class="material-icons-round">person</span>
        </div>
        <div class="assistant-bubble">
            <p>${message}</p>
        </div>
    `;
    chatArea.appendChild(userMsg);
    
    input.value = '';
    chatArea.scrollTop = chatArea.scrollHeight;
    
    // Simulate bot response
    setTimeout(() => {
        const botMsg = document.createElement('div');
        botMsg.className = 'assistant-message bot';
        botMsg.innerHTML = `
            <div class="assistant-avatar">
                <span class="material-icons-round">smart_toy</span>
            </div>
            <div class="assistant-bubble">
                <p>I'm a demo assistant. In production, I would help with claims procedures and policy questions. Your question: "${message}"</p>
            </div>
        `;
        chatArea.appendChild(botMsg);
        chatArea.scrollTop = chatArea.scrollHeight;
    }, 1000);
}

function handleAssistantEnter(event) {
    if (event.key === 'Enter') {
        sendAssistantMessage();
    }
}

console.log('%cDesigned by Kunal Baghel | 2026', 'color: #5f6368; font-size: 12px;');