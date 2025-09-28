// AI Academic Advisor Chatbot JavaScript
class AIAdvisorChatbot {
    constructor() {
        this.currentStudentId = null;
        this.chatHistory = [];
        this.isLoading = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.setWelcomeTime();
    }

    initializeElements() {
        // Main sections
        this.loginSection = document.getElementById('loginSection');
        this.chatContainer = document.getElementById('chatContainer');
        this.studentInfo = document.getElementById('studentInfo');
        this.currentStudentIdElement = document.getElementById('currentStudentId');
        
        // Login elements
        this.studentIdInput = document.getElementById('studentIdInput');
        this.loginBtn = document.getElementById('loginBtn');
        this.sampleBtns = document.querySelectorAll('.sample-btn');
        
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        this.quickBtns = document.querySelectorAll('.quick-btn');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Loading and modal elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.closeErrorModal = document.getElementById('closeErrorModal');
        this.errorModalOk = document.getElementById('errorModalOk');
        this.changeStudentBtn = document.getElementById('changeStudentBtn');
    }

    attachEventListeners() {
        // Login events
        this.loginBtn.addEventListener('click', () => this.handleLogin());
        this.studentIdInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleLogin();
        });
        
        // Sample student ID buttons
        this.sampleBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.studentIdInput.value = e.target.dataset.id;
                this.handleLogin();
            });
        });
        
        // Chat events
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });
        
        // Quick action buttons
        this.quickBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const question = e.target.dataset.question;
                this.messageInput.value = question;
                this.handleSendMessage();
            });
        });
        
        // Other controls
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.changeStudentBtn.addEventListener('click', () => this.changeStudent());
        
        // Modal events
        this.closeErrorModal.addEventListener('click', () => this.hideErrorModal());
        this.errorModalOk.addEventListener('click', () => this.hideErrorModal());
        
        // Close modal on outside click
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal) this.hideErrorModal();
        });
    }

    setWelcomeTime() {
        const welcomeTime = document.getElementById('welcomeTime');
        if (welcomeTime) {
            welcomeTime.textContent = this.getCurrentTime();
        }
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    async handleLogin() {
        const studentId = this.studentIdInput.value.trim();
        
        if (!studentId) {
            this.showError('Please enter a student ID');
            return;
        }

        this.showLoading('Loading your academic profile...');
        
        try {
            // Validate student exists in database
            const response = await fetch(`/api/chatbot/student-profile/${studentId}`);
            
            if (response.ok) {
                const profile = await response.json();
                this.currentStudentId = studentId;
                this.currentStudentIdElement.textContent = studentId;
                this.showChatInterface();
                this.addBotMessage(`Welcome back, ${studentId}! I have your academic profile loaded and ready to help.`);
            } else if (response.status === 404) {
                this.showError(`Student ID "${studentId}" not found in the database. Please use a valid student ID from the system.`);
            } else {
                this.showError('Failed to validate student. Please try again.');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showError('Connection error. Please check if the server is running.');
        } finally {
            this.hideLoading();
        }
    }

    async handleSendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message || this.isLoading) return;
        
        if (!this.currentStudentId) {
            this.showError('Please login first');
            return;
        }

        // Add user message to chat
        this.addUserMessage(message);
        this.messageInput.value = '';
        this.setSendButtonState(true);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/chatbot/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    student_id: this.currentStudentId,
                    question: message
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.hideTypingIndicator();
                this.addBotMessage(data.answer);
                
                // Store in chat history
                this.chatHistory.push({
                    user: message,
                    bot: data.answer,
                    timestamp: new Date(),
                    confidence: data.confidence
                });
            } else {
                this.hideTypingIndicator();
                const errorData = await response.json();
                this.addBotMessage(`Sorry, I encountered an error: ${errorData.detail || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addBotMessage('Sorry, I\'m having trouble connecting. Please try again in a moment.');
        } finally {
            this.setSendButtonState(false);
        }
    }

    addUserMessage(message) {
        const messageElement = this.createMessageElement('user', message);
        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    }

    addBotMessage(message) {
        const messageElement = this.createMessageElement('bot', message);
        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    }

    createMessageElement(type, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const text = document.createElement('div');
        text.className = 'message-text';
        text.innerHTML = this.formatMessage(message);
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = this.getCurrentTime();
        
        content.appendChild(text);
        content.appendChild(time);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        return messageDiv;
    }

    formatMessage(message) {
        // Convert markdown-like formatting to HTML
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/•/g, '&bull;')
            .replace(/→/g, '&rarr;')
            .replace(/✅/g, '<span style="color: #28a745;">✅</span>')
            .replace(/⚠️/g, '<span style="color: #ffc107;">⚠️</span>')
            .replace(/💡/g, '<span style="color: #17a2b8;">💡</span>')
            .replace(/🎓/g, '<span style="color: #6f42c1;">🎓</span>')
            .replace(/📈/g, '<span style="color: #20c997;">📈</span>')
            .replace(/📊/g, '<span style="color: #fd7e14;">📊</span>');
    }

    showChatInterface() {
        this.loginSection.style.display = 'none';
        this.chatContainer.style.display = 'flex';
        this.studentInfo.style.display = 'flex';
        this.messageInput.focus();
    }

    changeStudent() {
        this.currentStudentId = null;
        this.chatHistory = [];
        this.studentInfo.style.display = 'none';
        this.chatContainer.style.display = 'none';
        this.loginSection.style.display = 'flex';
        this.studentIdInput.value = '';
        this.studentIdInput.focus();
        
        // Clear chat messages except welcome message
        const messages = this.chatMessages.querySelectorAll('.message');
        messages.forEach((msg, index) => {
            if (index > 0) { // Keep the first welcome message
                msg.remove();
            }
        });
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.chatHistory = [];
            this.chatMessages.innerHTML = `
                <div class="message bot-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-text">
                            Chat cleared! How can I help you today?
                        </div>
                        <div class="message-time">${this.getCurrentTime()}</div>
                    </div>
                </div>
            `;
        }
    }

    showLoading(message = 'Loading...') {
        this.isLoading = true;
        this.loadingOverlay.style.display = 'flex';
        if (message) {
            this.loadingOverlay.querySelector('p').textContent = message;
        }
    }

    hideLoading() {
        this.isLoading = false;
        this.loadingOverlay.style.display = 'none';
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    setSendButtonState(disabled) {
        this.sendBtn.disabled = disabled;
        this.messageInput.disabled = disabled;
        
        if (disabled) {
            this.sendBtn.style.opacity = '0.6';
            this.messageInput.placeholder = 'AI Advisor is thinking...';
        } else {
            this.sendBtn.style.opacity = '1';
            this.messageInput.placeholder = 'Ask me about courses, prerequisites, or academic planning...';
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'flex';
    }

    hideErrorModal() {
        this.errorModal.style.display = 'none';
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new AIAdvisorChatbot();
});

// Handle page visibility change to refocus input
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.aiAdvisor) {
        const messageInput = document.getElementById('messageInput');
        if (messageInput && messageInput.offsetParent !== null) {
            messageInput.focus();
        }
    }
});

// Export for global access
window.AIAdvisorChatbot = AIAdvisorChatbot;
