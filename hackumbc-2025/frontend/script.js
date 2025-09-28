// Academic Risk Predictor Frontend JavaScript

class AcademicRiskPredictor {
    constructor() {
        this.apiUrl = '';  // Use relative URLs since frontend is served from same server
        this.form = document.getElementById('predictionForm');
        this.loading = document.getElementById('loading');
        this.results = document.getElementById('results');
        this.error = document.getElementById('error');
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });
    }

    async handlePrediction() {
        const studentId = document.getElementById('studentId').value.trim();
        const courseId = document.getElementById('courseId').value.trim();

        if (!studentId || !courseId) {
            this.showError('Please enter both Student ID and Course ID');
            return;
        }

        this.showLoading();
        this.hideResults();
        this.hideError();

        try {
            const prediction = await this.getPrediction(studentId, courseId);
            this.displayResults(prediction);
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to get prediction. Please check if the server is running and try again.');
        }
    }

    async getPrediction(studentId, courseId) {
        const response = await fetch(`${this.apiUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_id: studentId,
                course_id: courseId
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Server error');
        }

        return await response.json();
    }

    displayResults(prediction) {
        const { prediction_result, confidence, risk_level, recommendations } = prediction;
        
        // Update risk level
        const riskLevelElement = document.getElementById('riskLevel');
        riskLevelElement.textContent = risk_level;
        riskLevelElement.className = `risk-level ${prediction_result === 1 ? 'low' : 'high'}`;
        
        // Update confidence score
        const confidenceElement = document.getElementById('confidenceScore');
        confidenceElement.textContent = `${(confidence * 100).toFixed(1)}%`;
        
        // Update recommendations
        const recommendationsElement = document.getElementById('recommendations');
        recommendationsElement.innerHTML = `
            <h4>Recommendations:</h4>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
        
        this.hideLoading();
        this.showResults();
    }

    showLoading() {
        this.loading.classList.remove('hidden');
        document.getElementById('predictBtn').disabled = true;
    }

    hideLoading() {
        this.loading.classList.add('hidden');
        document.getElementById('predictBtn').disabled = false;
    }

    showResults() {
        this.results.classList.remove('hidden');
    }

    hideResults() {
        this.results.classList.add('hidden');
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.error.classList.remove('hidden');
        this.hideLoading();
    }

    hideError() {
        this.error.classList.add('hidden');
    }
}

// Sample data for demo purposes
class DemoData {
    static getSampleStudents() {
        return [
            'ZO28124', 'XN08759', 'EY56522', 'PX26385', 'XE28807',
            'OU90944', 'EL31170', 'KH74592', 'NA63594', 'STUDENT123'
        ];
    }

    static getSampleCourses() {
        return [
            'CSEE 200', 'CSLL 100-6', 'BUUU 100', 'BGGG 100', 'BLLL 100',
            'BKKK 200', 'CSJJ 100-4', 'CSYY 200', 'BJJJ 100', 'CSHH 100'
        ];
    }

    static populateSampleData() {
        const students = this.getSampleStudents();
        const courses = this.getSampleCourses();
        
        document.getElementById('studentId').value = students[Math.floor(Math.random() * students.length)];
        document.getElementById('courseId').value = courses[Math.floor(Math.random() * courses.length)];
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new AcademicRiskPredictor();
    
    // Add sample data button for demo
    const form = document.getElementById('predictionForm');
    const sampleButton = document.createElement('button');
    sampleButton.type = 'button';
    sampleButton.textContent = '🎲 Use Sample Data';
    sampleButton.className = 'sample-btn';
    sampleButton.style.cssText = `
        background: #48bb78;
        margin-top: 10px;
        width: 100%;
        padding: 10px;
        font-size: 14px;
    `;
    sampleButton.addEventListener('click', () => {
        DemoData.populateSampleData();
    });
    
    form.appendChild(sampleButton);
    
    // Add some interactive features
    const inputs = form.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    });
});

// Add some visual feedback
document.addEventListener('DOMContentLoaded', () => {
    // Add typing animation effect
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            this.style.borderColor = '#667eea';
            setTimeout(() => {
                this.style.borderColor = '#e2e8f0';
            }, 1000);
        });
    });
    
    // Add success animation for form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', () => {
        const button = document.getElementById('predictBtn');
        button.style.background = 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)';
        setTimeout(() => {
            button.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        }, 2000);
    });
});
