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

function goToMentorship() {
    // Store student ID in session storage for persistence
    if (currentStudentId) {
        sessionStorage.setItem('studentId', currentStudentId);
        window.location.href = `/mentorship?student_id=${currentStudentId}`;
    } else {
        window.location.href = '/mentorship';
    }
}