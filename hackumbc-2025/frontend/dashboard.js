// Dashboard JavaScript - Real-time Academic Analytics Dashboard

class AcademicDashboard {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.charts = {};
        this.updateInterval = null;
        this.isLoading = false;
        this.currentStudentId = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.startRealTimeUpdates();
        this.setupCharts();
    }

    setupEventListeners() {
        // Search functionality
        const studentSearch = document.getElementById('studentSearch');
        studentSearch.addEventListener('input', this.debounce(this.handleStudentSearch.bind(this), 300));

        // Refresh button
        const refreshBtn = document.getElementById('refreshData');
        refreshBtn.addEventListener('click', () => this.loadInitialData());

        // Student analysis
        const analyzeBtn = document.getElementById('analyzeStudentBtn');
        analyzeBtn.addEventListener('click', this.analyzeStudent.bind(this));

        const studentIdInput = document.getElementById('studentIdInput');
        studentIdInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.analyzeStudent();
            }
        });

        // Study groups
        const generateGroupsBtn = document.getElementById('generateGroupsBtn');
        generateGroupsBtn.addEventListener('click', this.generateStudyGroups.bind(this));

        // Chart filters
        const performanceFilter = document.getElementById('performanceFilter');
        performanceFilter.addEventListener('change', () => this.updatePerformanceChart());

        const courseFilter = document.getElementById('courseFilter');
        courseFilter.addEventListener('change', () => this.updateCourseSuccessChart());

        // Error modal
        const closeErrorBtn = document.getElementById('closeErrorBtn');
        closeErrorBtn.addEventListener('click', () => this.hideErrorModal());
    }

    async loadInitialData() {
        if (this.isLoading) return;
        
        this.showLoading();
        try {
            await Promise.all([
                this.loadOverviewData(),
                this.loadChartData(),
                this.loadCoursesForStudyGroups()
            ]);
            this.updateLastUpdatedTime();
            this.addUpdate('Dashboard data refreshed successfully', 'success');
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load dashboard data. Please try again.');
            this.addUpdate('Error loading dashboard data', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadOverviewData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/dashboard/overview`);
            if (!response.ok) throw new Error('Failed to load overview data');
            
            const data = await response.json();
            this.updateOverviewCards(data);
        } catch (error) {
            console.error('Error loading overview data:', error);
            throw error;
        }
    }

    async loadChartData() {
        try {
            const [atRiskData, predictionsData] = await Promise.all([
                fetch(`${this.apiBaseUrl}/dashboard/at-risk-students`).then(r => r.json()),
                fetch(`${this.apiBaseUrl}/dashboard/predictions`).then(r => r.json())
            ]);

            this.updatePerformanceChart(atRiskData);
            this.updateCourseSuccessChart(predictionsData);
            this.updateLearningStyleChart();
            this.updateRiskFactorsChart(predictionsData);
        } catch (error) {
            console.error('Error loading chart data:', error);
            throw error;
        }
    }

    async loadCoursesForStudyGroups() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/courses`);
            if (!response.ok) throw new Error('Failed to load courses');
            
            const data = await response.json();
            this.populateCourseSelect(data.courses);
        } catch (error) {
            console.error('Error loading courses:', error);
        }
    }

    updateOverviewCards(data) {
        document.getElementById('totalStudents').textContent = data.total_students || '--';
        document.getElementById('totalCourses').textContent = data.total_courses || '--';
        document.getElementById('successRate').textContent = `${data.success_rate || 0}%`;
        document.getElementById('atRiskCount').textContent = data.at_risk_students || '--';
    }

    setupCharts() {
        // Performance Chart
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        this.charts.performance = new Chart(performanceCtx, {
            type: 'doughnut',
            data: {
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#e74c3c', '#f39c12', '#27ae60'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });

        // Course Success Chart
        const courseSuccessCtx = document.getElementById('courseSuccessChart').getContext('2d');
        this.charts.courseSuccess = new Chart(courseSuccessCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Success Rate (%)',
                    data: [],
                    backgroundColor: '#3498db',
                    borderColor: '#2980b9',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        // Learning Style Chart
        const learningStyleCtx = document.getElementById('learningStyleChart').getContext('2d');
        this.charts.learningStyle = new Chart(learningStyleCtx, {
            type: 'pie',
            data: {
                labels: ['Visual', 'Auditory', 'Kinesthetic', 'Reading-Writing'],
                datasets: [{
                    data: [35, 25, 30, 10],
                    backgroundColor: ['#9b59b6', '#3498db', '#e67e22', '#2ecc71'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });

        // Risk Factors Chart
        const riskFactorsCtx = document.getElementById('riskFactorsChart').getContext('2d');
        this.charts.riskFactors = new Chart(riskFactorsCtx, {
            type: 'horizontalBar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Percentage',
                    data: [],
                    backgroundColor: '#e74c3c',
                    borderColor: '#c0392b',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    updatePerformanceChart(atRiskData) {
        if (!atRiskData) return;
        
        const data = [
            atRiskData.high_risk || 0,
            atRiskData.medium_risk || 0,
            atRiskData.low_risk || 0
        ];
        
        this.charts.performance.data.datasets[0].data = data;
        this.charts.performance.update();
    }

    updateCourseSuccessChart(predictionsData) {
        if (!predictionsData || !predictionsData.success_probability_by_course) return;
        
        const courses = predictionsData.success_probability_by_course.slice(0, 8);
        const labels = courses.map(c => c.course);
        const data = courses.map(c => c.probability);
        
        this.charts.courseSuccess.data.labels = labels;
        this.charts.courseSuccess.data.datasets[0].data = data;
        this.charts.courseSuccess.update();
    }

    updateLearningStyleChart() {
        // This would typically come from the API, but we'll use sample data
        const data = [35, 25, 30, 10];
        this.charts.learningStyle.data.datasets[0].data = data;
        this.charts.learningStyle.update();
    }

    updateRiskFactorsChart(predictionsData) {
        if (!predictionsData || !predictionsData.risk_factors) return;
        
        const factors = predictionsData.risk_factors;
        const labels = factors.map(f => f.factor);
        const data = factors.map(f => f.percentage);
        
        this.charts.riskFactors.data.labels = labels;
        this.charts.riskFactors.data.datasets[0].data = data;
        this.charts.riskFactors.update();
    }

    async analyzeStudent() {
        const studentId = document.getElementById('studentIdInput').value.trim();
        if (!studentId) {
            this.showError('Please enter a student ID');
            return;
        }

        this.showLoading();
        try {
            const studentData = await this.getStudentData(studentId);
            this.displayStudentAnalysis(studentData);
            this.currentStudentId = studentId;
            this.addUpdate(`Analyzed student ${studentId}`, 'info');
        } catch (error) {
            console.error('Error analyzing student:', error);
            this.showError(`Student ${studentId} not found or error occurred`);
            this.addUpdate(`Error analyzing student ${studentId}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async getStudentData(studentId) {
        const response = await fetch(`${this.apiBaseUrl}/api/student/${studentId}`);
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Student not found');
            }
            throw new Error('Failed to load student data');
        }
        return await response.json();
    }

    displayStudentAnalysis(studentData) {
        const content = document.getElementById('studentDetailsContent');
        
        const analysisHTML = `
            <div class="student-analysis">
                <div class="student-info">
                    <h4>Student Information</h4>
                    <div class="info-item">
                        <span class="info-label">Student ID:</span>
                        <span class="info-value">${studentData.id}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Name:</span>
                        <span class="info-value">${studentData.name}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Major:</span>
                        <span class="info-value">${studentData.major}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Year:</span>
                        <span class="info-value">${studentData.year}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">GPA:</span>
                        <span class="info-value">${studentData.gpa}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Risk Level:</span>
                        <span class="info-value">
                            <span class="risk-indicator ${studentData.risk}">${studentData.risk}</span>
                        </span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Credits Completed:</span>
                        <span class="info-value">${studentData.creditsCompleted}</span>
                    </div>
                </div>
                
                <div class="student-info">
                    <h4>Current Courses</h4>
                    ${studentData.coursesThisSemester && studentData.coursesThisSemester.length > 0 
                        ? studentData.coursesThisSemester.map(course => `
                            <div class="info-item">
                                <span class="info-label">${course.code}:</span>
                                <span class="info-value">${course.name} (${course.grade}%)</span>
                            </div>
                        `).join('')
                        : '<p>No current courses</p>'
                    }
                    
                    <h4 style="margin-top: 1rem;">Recommendations</h4>
                    ${studentData.recommendations && studentData.recommendations.length > 0 
                        ? `<ul style="margin-top: 0.5rem;">${studentData.recommendations.map(rec => `<li>${rec}</li>`).join('')}</ul>`
                        : '<p>No specific recommendations</p>'
                    }
                </div>
            </div>
        `;
        
        content.innerHTML = analysisHTML;
        content.classList.add('fade-in');
    }

    async generateStudyGroups() {
        const courseId = document.getElementById('courseSelect').value;
        if (!courseId) {
            this.showError('Please select a course');
            return;
        }

        this.showLoading();
        try {
            const response = await fetch(`${this.apiBaseUrl}/study-groups`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    course_id: courseId,
                    min_group_size: 3,
                    max_group_size: 5
                })
            });

            if (!response.ok) throw new Error('Failed to generate study groups');
            
            const groups = await response.json();
            this.displayStudyGroups(groups);
            this.addUpdate(`Generated study groups for ${courseId}`, 'success');
        } catch (error) {
            console.error('Error generating study groups:', error);
            this.showError('Failed to generate study groups');
            this.addUpdate(`Error generating study groups for ${courseId}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayStudyGroups(groups) {
        const content = document.getElementById('studyGroupsContent');
        
        if (!groups || groups.length === 0) {
            content.innerHTML = '<p>No study groups available for this course.</p>';
            return;
        }

        const groupsHTML = groups.map(group => `
            <div class="study-group-card">
                <div class="study-group-header">
                    <span class="group-name">${group.group_id}</span>
                    <span class="compatibility-score">${Math.round(group.avg_compatibility)}% Compatible</span>
                </div>
                <div class="group-members">
                    ${group.members.map(member => `
                        <span class="member-tag">${member.name}</span>
                    `).join('')}
                </div>
                <div class="group-details">
                    <div class="group-detail-item">
                        <span class="group-detail-label">Course:</span>
                        <span class="group-detail-value">${group.course_name}</span>
                    </div>
                    <div class="group-detail-item">
                        <span class="group-detail-label">Meeting Time:</span>
                        <span class="group-detail-value">${group.recommended_meeting_time}</span>
                    </div>
                    <div class="group-detail-item">
                        <span class="group-detail-label">Group Size:</span>
                        <span class="group-detail-value">${group.group_size} members</span>
                    </div>
                    <div class="group-detail-item">
                        <span class="group-detail-label">Diversity:</span>
                        <span class="group-detail-value">${Math.round(group.learning_style_diversity * 100)}%</span>
                    </div>
                </div>
            </div>
        `).join('');

        content.innerHTML = groupsHTML;
        content.classList.add('fade-in');
    }

    populateCourseSelect(courses) {
        const select = document.getElementById('courseSelect');
        select.innerHTML = '<option value="">Select a course...</option>';
        
        courses.forEach(course => {
            const option = document.createElement('option');
            option.value = course;
            option.textContent = course;
            select.appendChild(option);
        });
    }

    async handleStudentSearch(event) {
        const query = event.target.value.trim();
        if (query.length < 3) return;

        try {
            // This would typically search through available students
            // For now, we'll just show a message
            this.addUpdate(`Searching for students matching "${query}"`, 'info');
        } catch (error) {
            console.error('Error searching students:', error);
        }
    }

    startRealTimeUpdates() {
        // Update data every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadInitialData();
        }, 30000);
    }

    addUpdate(message, type = 'info') {
        const updatesList = document.getElementById('updatesList');
        const updateItem = document.createElement('div');
        updateItem.className = 'update-item';
        
        const icon = type === 'error' ? 'fa-exclamation-circle' : 
                    type === 'success' ? 'fa-check-circle' : 
                    type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';
        
        const color = type === 'error' ? '#e74c3c' : 
                     type === 'success' ? '#27ae60' : 
                     type === 'warning' ? '#f39c12' : '#3498db';
        
        updateItem.innerHTML = `
            <i class="fas ${icon}" style="color: ${color}"></i>
            <span>${message}</span>
            <span class="update-time">${new Date().toLocaleTimeString()}</span>
        `;
        
        updatesList.insertBefore(updateItem, updatesList.firstChild);
        
        // Keep only last 10 updates
        while (updatesList.children.length > 10) {
            updatesList.removeChild(updatesList.lastChild);
        }
    }

    updateLastUpdatedTime() {
        const timeElement = document.getElementById('lastUpdateTime');
        timeElement.textContent = new Date().toLocaleTimeString();
    }

    showLoading() {
        this.isLoading = true;
        document.getElementById('loadingOverlay').classList.remove('hidden');
    }

    hideLoading() {
        this.isLoading = false;
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorModal').classList.remove('hidden');
    }

    hideErrorModal() {
        document.getElementById('errorModal').classList.add('hidden');
    }

    debounce(func, wait) {
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
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AcademicDashboard();
});

// Handle page visibility changes to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, could pause updates here
    } else {
        // Page is visible, resume updates
        if (window.dashboard) {
            window.dashboard.loadInitialData();
        }
    }
});

// Export for global access
window.AcademicDashboard = AcademicDashboard;
