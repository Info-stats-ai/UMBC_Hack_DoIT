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
});

function openRiskPrediction() {
    // Navigate to the new risk assessment page with student ID
    if (currentStudentId) {
        window.location.href = `/risk-assessment?studentId=${encodeURIComponent(currentStudentId)}`;
    } else {
        alert('Student ID not found. Please return to the home page.');
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