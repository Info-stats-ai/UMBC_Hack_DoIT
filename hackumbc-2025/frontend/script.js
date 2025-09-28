// UMBC Academic Portal - Main Page JavaScript

class StudentLogin {
    constructor() {
        this.form = document.getElementById('studentLoginForm');
        this.studentIdInput = document.getElementById('studentId');
        this.continueBtn = document.getElementById('continueBtn');
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        if (this.form) {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleStudentLogin();
            });
        }
    }

    handleStudentLogin() {
        const studentId = this.studentIdInput.value.trim();
        
        if (!studentId) {
            this.showError('Please enter your Student ID');
            return;
        }

        // Validate student ID format (basic validation)
        if (!this.isValidStudentId(studentId)) {
            this.showError('Please enter a valid Student ID format (e.g., ZO28124)');
            return;
        }

        // Navigate to student options page
        this.navigateToOptions(studentId);
    }

    isValidStudentId(studentId) {
        // Basic validation - should be alphanumeric and reasonable length
        return /^[A-Z0-9]{5,10}$/i.test(studentId);
    }

    navigateToOptions(studentId) {
        // Navigate to student options page with student ID as parameter
        window.location.href = `/student-options?studentId=${encodeURIComponent(studentId)}`;
    }

    showError(message) {
        // Create or update error message
        let errorDiv = document.getElementById('error');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'error';
            errorDiv.className = 'error';
            errorDiv.style.cssText = `
                background: #fed7d7;
                color: #c53030;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                text-align: center;
            `;
            this.form.appendChild(errorDiv);
        }
        
        errorDiv.innerHTML = `<strong>❌ Error:</strong> ${message}`;
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            if (errorDiv && errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Sample Student IDs for demo purposes
class DemoData {
    static getSampleStudents() {
        return [
            'ZO28124', 'XN08759', 'EY56522', 'PX26385', 'XE28807',
            'OU90944', 'EL31170', 'KH74592', 'NA63594', 'STUDENT123'
        ];
    }

    static populateSampleData() {
        const students = this.getSampleStudents();
        const randomStudent = students[Math.floor(Math.random() * students.length)];
        
        const studentIdInput = document.getElementById('studentId');
        if (studentIdInput) {
            studentIdInput.value = randomStudent;
        }
    }
}

// Global function for accessing risk prediction
function accessRiskPrediction() {
    const studentIdInput = document.getElementById('studentId');
    const studentId = studentIdInput ? studentIdInput.value.trim() : '';
    
    if (!studentId) {
        alert('Please enter your Student ID first to access Risk Prediction');
        studentIdInput.focus();
        return;
    }
    
    // Validate student ID format
    if (!/^[A-Z0-9]{5,10}$/i.test(studentId)) {
        alert('Please enter a valid Student ID format (e.g., ZO28124)');
        studentIdInput.focus();
        return;
    }
    
    // Navigate to risk prediction page with student ID
    window.location.href = `/risk-prediction?studentId=${encodeURIComponent(studentId)}`;
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new StudentLogin();
    
    // Add sample data button for demo
    const form = document.getElementById('studentLoginForm');
    if (form) {
        const sampleButton = document.createElement('button');
        sampleButton.type = 'button';
        sampleButton.textContent = '🎲 Use Sample Student ID';
        sampleButton.className = 'sample-btn';
        sampleButton.style.cssText = `
            background: #48bb78;
            margin-top: 10px;
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            transition: background 0.3s ease;
        `;
        
        sampleButton.addEventListener('click', () => {
            DemoData.populateSampleData();
        });
        
        sampleButton.addEventListener('mouseenter', () => {
            sampleButton.style.background = '#38a169';
        });
        
        sampleButton.addEventListener('mouseleave', () => {
            sampleButton.style.background = '#48bb78';
        });
        
        form.appendChild(sampleButton);
    }
    
    // Add some interactive features
    const studentIdInput = document.getElementById('studentId');
    if (studentIdInput) {
        studentIdInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
        
        // Add typing animation effect
        studentIdInput.addEventListener('input', function() {
            this.style.borderColor = '#667eea';
            setTimeout(() => {
                this.style.borderColor = '#e2e8f0';
            }, 1000);
        });
    }
    
    // Add success animation for form submission
    const continueBtn = document.getElementById('continueBtn');
    if (continueBtn) {
        continueBtn.addEventListener('click', () => {
            continueBtn.style.background = 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)';
            setTimeout(() => {
                continueBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            }, 2000);
        });
    }
});
