// Student Options Page JavaScript
let currentStudentId = '';

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Get student ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    currentStudentId = urlParams.get('studentId') || '';
    
    // Display student ID
    const studentDisplayElement = document.getElementById('studentDisplayId');
    if (studentDisplayElement) {
        studentDisplayElement.textContent = currentStudentId || 'Student';
    }
    
    // Initialize form handlers
    initializeForms();
});

function initializeForms() {
    // Risk prediction form handler
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handleRiskPrediction);
    }
}

function openRiskPrediction() {
    const optionsSection = document.querySelector('.options-section');
    const riskSection = document.getElementById('riskPredictionSection');
    
    if (optionsSection && riskSection) {
        optionsSection.classList.add('hidden');
        riskSection.classList.remove('hidden');
    }
}

function closeRiskPrediction() {
    const optionsSection = document.querySelector('.options-section');
    const riskSection = document.getElementById('riskPredictionSection');

    if (optionsSection && riskSection) {
        optionsSection.classList.remove('hidden');
        riskSection.classList.add('hidden');

        // Clear any previous results
        clearResults();
    }
}

function openCoursePlanning() {
    // Navigate to course planning page with student ID
    if (currentStudentId) {
        window.location.href = `/course-planning?studentId=${encodeURIComponent(currentStudentId)}`;
    } else {
        alert('Student ID not found. Please return to the home page.');
    }
}

function openAcademicAdvisor() {
    console.log('openAcademicAdvisor called');
    
    const optionsSection = document.querySelector('.options-section');
    const advisorSection = document.getElementById('academicAdvisorSection');
    
    console.log('optionsSection:', optionsSection);
    console.log('advisorSection:', advisorSection);
    
    if (optionsSection && advisorSection) {
        console.log('Both sections found, switching views');
        optionsSection.classList.add('hidden');
        advisorSection.classList.remove('hidden');
        
        // Initialize the chat interface
        initializeChatInterface();
    } else {
        console.error('Could not find required sections');
        alert('Error: Could not find required sections');
    }
}

function closeAcademicAdvisor() {
    const optionsSection = document.querySelector('.options-section');
    const advisorSection = document.getElementById('academicAdvisorSection');

    if (optionsSection && advisorSection) {
        optionsSection.classList.remove('hidden');
        advisorSection.classList.add('hidden');
    }
}

async function handleRiskPrediction(event) {
    event.preventDefault();
    
    const courseId = document.getElementById('courseId').value;
    
    if (!currentStudentId) {
        showError('Student ID not found. Please return to the main page.');
        return;
    }
    
    if (!courseId) {
        showError('Please enter a course ID.');
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_id: currentStudentId,
                course_id: courseId
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(`Prediction failed: ${error.message}`);
    }
}

function displayResults(data) {
    hideLoading();
    
    const resultsDiv = document.getElementById('results');
    const riskLevelDiv = document.getElementById('riskLevel');
    const confidenceScoreDiv = document.getElementById('confidenceScore');
    const recommendationsDiv = document.getElementById('recommendations');
    
    if (!resultsDiv || !riskLevelDiv || !confidenceScoreDiv || !recommendationsDiv) {
        showError('Error displaying results');
        return;
    }
    
    // Display risk level
    const riskClass = data.prediction_result === 1 ? 'low-risk' : 'high-risk';
    riskLevelDiv.className = `risk-level ${riskClass}`;
    riskLevelDiv.innerHTML = `
        <div class="risk-icon">${data.prediction_result === 1 ? '✅' : '⚠️'}</div>
        <div class="risk-text">
            <h4>${data.risk_level}</h4>
            <p>Student: ${data.student_id} | Course: ${data.course_id}</p>
        </div>
    `;
    
    // Display confidence score
    confidenceScoreDiv.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    
    // Display recommendations
    recommendationsDiv.innerHTML = `
        <h5>Recommendations:</h5>
        <ul>
            ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
        <div class="probability-info">
            <h5>Probability Breakdown:</h5>
            <div class="probability-bars">
                <div class="prob-bar">
                    <span class="prob-label">High Risk:</span>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill high-risk" style="width: ${data.probability.high_risk * 100}%"></div>
                        <span class="prob-percentage">${(data.probability.high_risk * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="prob-bar">
                    <span class="prob-label">Low Risk:</span>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill low-risk" style="width: ${data.probability.low_risk * 100}%"></div>
                        <span class="prob-percentage">${(data.probability.low_risk * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsDiv.classList.remove('hidden');
}

function showLoading() {
    const loadingDiv = document.getElementById('loading');
    if (loadingDiv) {
        loadingDiv.classList.remove('hidden');
    }
    hideResults();
    hideError();
}

function hideLoading() {
    const loadingDiv = document.getElementById('loading');
    if (loadingDiv) {
        loadingDiv.classList.add('hidden');
    }
}

function showError(message) {
    hideLoading();
    hideResults();
    
    const errorDiv = document.getElementById('error');
    const errorMessageDiv = document.getElementById('errorMessage');
    
    if (errorDiv && errorMessageDiv) {
        errorMessageDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
}

function hideError() {
    const errorDiv = document.getElementById('error');
    if (errorDiv) {
        errorDiv.classList.add('hidden');
    }
}

function hideResults() {
    const resultsDiv = document.getElementById('results');
    if (resultsDiv) {
        resultsDiv.classList.add('hidden');
    }
}

function clearResults() {
    hideResults();
    hideError();
    hideLoading();
    
    // Clear form
    const courseIdInput = document.getElementById('courseId');
    if (courseIdInput) {
        courseIdInput.value = '';
    }
}

// AI Advisory Chat Interface
function initializeChatInterface() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const quickBtns = document.querySelectorAll('.quick-btn');
    
    // Set current time for welcome message
    const welcomeTime = document.getElementById('welcomeTime');
    if (welcomeTime) {
        welcomeTime.textContent = new Date().toLocaleTimeString();
    }
    
    // Send message functionality
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    if (messageInput) {
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    // Clear chat functionality
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChat);
    }
    
    // Quick action buttons
    quickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            if (question && messageInput) {
                messageInput.value = question;
                sendMessage();
            }
        });
    });
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Call the real AI advisory API
        const response = await fetch('/api/chatbot/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_id: currentStudentId,
                question: message
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get AI response');
        }
        
        const data = await response.json();
        
        // Debug: Log the response data
        console.log('AI Advisory Response:', data);
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Combine AI response with recommendations for a single, comprehensive message
        let fullResponse = data.answer;
        
        // Add detailed recommendations if available
        if (data.recommendations && data.recommendations.length > 0) {
            const recommendationsText = formatRecommendations(data.recommendations);
            if (recommendationsText.trim()) {
                fullResponse += '\n\n' + recommendationsText;
            }
        }
        
        // Add context note if AI response seems generic
        if (data.recommendations && data.recommendations.length > 0 && 
            (data.answer.includes('RateMyProfessor') || data.answer.includes('general advice'))) {
            fullResponse += '\n\n💡 **Note:** The above recommendations are based on your actual academic data and similar students\' course patterns. Use this detailed information to make informed decisions about your course selection.';
        }
        
        // Display the combined response as a single message
        addMessage(fullResponse, 'bot');
        
    } catch (error) {
        console.error('AI Advisory error:', error);
        hideTypingIndicator();
        
        // Show error message
        addMessage(`Sorry, I encountered an error: ${error.message}. Please try again or contact support.`, 'bot');
    }
}

function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = sender === 'bot' ? 'fas fa-robot' : 'fas fa-user';
    const currentTime = new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="${avatar}"></i>
        </div>
        <div class="message-content">
            <div class="message-text">${text}</div>
            <div class="message-time">${currentTime}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'flex';
    }
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        // Keep only the welcome message
        const welcomeMessage = chatMessages.querySelector('.bot-message');
        chatMessages.innerHTML = '';
        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }
    }
}

function formatRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        return '';
    }
    
    let formatted = '📋 **Course Recommendations:**\n\n';
    
    recommendations.forEach((rec, index) => {
        formatted += `${index + 1}. **${rec.course_id}** - ${rec.course_name || 'Course Name Not Available'}\n`;
        
        // Credits
        if (rec.credits) {
            formatted += `   • Credits: ${rec.credits}\n`;
        }
        
        // Difficulty
        if (rec.difficulty) {
            formatted += `   • Difficulty: ${rec.difficulty}\n`;
        }
        
        // Instruction Mode
        if (rec.instruction_mode) {
            formatted += `   • Mode: ${rec.instruction_mode}\n`;
        }
        
        // Description
        if (rec.description) {
            formatted += `   • Description: ${rec.description.substring(0, 100)}${rec.description.length > 100 ? '...' : ''}\n`;
        }
        
        // Availability Status
        if (rec.is_available !== undefined) {
            formatted += `   • Status: ${rec.is_available ? '✅ Available' : '❌ Prerequisites Required'}\n`;
        }
        
        // Missing Prerequisites
        if (rec.missing_prerequisites && rec.missing_prerequisites.length > 0) {
            formatted += `   • Missing Prerequisites: ${rec.missing_prerequisites.join(', ')}\n`;
        }
        
        // Prerequisites (if available)
        if (rec.prerequisites && rec.prerequisites.length > 0) {
            formatted += `   • Prerequisites: ${rec.prerequisites.join(', ')}\n`;
        }
        
        // Faculty Options
        if (rec.faculty_options && rec.faculty_options.length > 0) {
            formatted += `   • Available Faculty: ${rec.faculty_options.map(f => f.name).join(', ')}\n`;
            if (rec.faculty_options.some(f => f.teachingStyle)) {
                const teachingStyles = rec.faculty_options
                    .filter(f => f.teachingStyle)
                    .map(f => `${f.name} (${f.teachingStyle})`)
                    .join(', ');
                if (teachingStyles) {
                    formatted += `   • Teaching Styles: ${teachingStyles}\n`;
                }
            }
        }
        
        // Leads To (Future Courses)
        if (rec.leads_to && rec.leads_to.length > 0) {
            formatted += `   • Leads to: ${rec.leads_to.join(', ')}\n`;
        }
        
        // Relevance Score
        if (rec.relevance_score) {
            formatted += `   • Relevance Score: ${rec.relevance_score}/100\n`;
        }
        
        // Recommendation Source
        if (rec.recommendation_source) {
            const source = rec.recommendation_source === 'similar_students' ? 'Similar Students' : 'Degree Requirements';
            formatted += `   • Recommended by: ${source}\n`;
        }
        
        formatted += '\n';
    });
    
    return formatted;
}

// Note: generateAIResponse function removed - now using real API calls to /api/chatbot/ask
