// Student Advisor Dashboard JavaScript

class AdvisorDashboard {
    constructor() {
        this.currentTab = 'overview';
        this.charts = {};
        this.sampleData = this.generateSampleData();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadOverviewData();
        this.initializeCharts();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Overview filters
        document.getElementById('semesterSelect')?.addEventListener('change', () => {
            this.loadOverviewData();
        });

        // Risk analysis filters
        document.getElementById('riskLevelFilter')?.addEventListener('change', () => {
            this.filterRiskStudents();
        });

        document.getElementById('departmentFilter')?.addEventListener('change', () => {
            this.filterRiskStudents();
        });

        // Course path generation
        document.getElementById('generatePathBtn')?.addEventListener('click', () => {
            this.generateOptimalCoursePath();
        });

        document.getElementById('majorSelect')?.addEventListener('change', () => {
            this.generateOptimalCoursePath();
        });

        // Study group generation
        document.getElementById('generateGroupsBtn')?.addEventListener('click', () => {
            this.generateStudyGroups();
        });

        // Prediction controls
        document.getElementById('refreshPredictionsBtn')?.addEventListener('click', () => {
            this.refreshPredictions();
        });

        document.getElementById('exportPredictionsBtn')?.addEventListener('click', () => {
            this.exportPredictions();
        });
    }

    switchTab(tabName) {
        // Update nav tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');

        this.currentTab = tabName;

        // Load tab-specific data
        switch(tabName) {
            case 'overview':
                this.loadOverviewData();
                break;
            case 'risk-analysis':
                this.loadRiskAnalysisData();
                break;
            case 'course-paths':
                this.generateOptimalCoursePath();
                break;
            case 'study-groups':
                this.generateStudyGroups();
                break;
            case 'predictions':
                this.loadPredictionData();
                break;
        }
    }

    generateSampleData() {
        return {
            students: [
                { id: 'ZO28124', name: 'Alex Johnson', major: 'Computer Science', gpa: 3.2, risk: 'high' },
                { id: 'XN08759', name: 'Sarah Chen', major: 'Mathematics', gpa: 3.8, risk: 'low' },
                { id: 'EY56522', name: 'Michael Brown', major: 'Biology', gpa: 2.9, risk: 'medium' },
                { id: 'PX26385', name: 'Emily Davis', major: 'Computer Science', gpa: 3.5, risk: 'low' },
                { id: 'XE28807', name: 'David Wilson', major: 'Engineering', gpa: 2.7, risk: 'high' },
                { id: 'OU90944', name: 'Lisa Garcia', major: 'Mathematics', gpa: 3.1, risk: 'medium' },
                { id: 'EL31170', name: 'James Miller', major: 'Biology', gpa: 3.6, risk: 'low' },
                { id: 'KH74592', name: 'Maria Rodriguez', major: 'Computer Science', gpa: 2.8, risk: 'high' }
            ],
            courses: [
                { code: 'CSEE 200', name: 'Introduction to Programming', credits: 3, difficulty: 'medium' },
                { code: 'MATH 151', name: 'Calculus I', credits: 4, difficulty: 'hard' },
                { code: 'BIOL 141', name: 'General Biology', credits: 4, difficulty: 'medium' },
                { code: 'CSEE 201', name: 'Data Structures', credits: 3, difficulty: 'hard' },
                { code: 'MATH 152', name: 'Calculus II', credits: 4, difficulty: 'hard' }
            ],
            studyGroups: [
                {
                    name: 'Group Alpha',
                    course: 'CSEE 200',
                    members: ['Student A (Visual)', 'Student B (Auditory)', 'Student C (Kinesthetic)'],
                    compatibility: 92,
                    avgGpa: 3.2,
                    meetingTime: 'Tue/Thu 2-4 PM'
                },
                {
                    name: 'Group Beta',
                    course: 'MATH 151',
                    members: ['Student D (Visual)', 'Student E (Reading)', 'Student F (Kinesthetic)'],
                    compatibility: 88,
                    avgGpa: 3.4,
                    meetingTime: 'Mon/Wed 3-5 PM'
                },
                {
                    name: 'Group Gamma',
                    course: 'BIOL 141',
                    members: ['Student G (Auditory)', 'Student H (Visual)', 'Student I (Reading)'],
                    compatibility: 85,
                    avgGpa: 3.1,
                    meetingTime: 'Fri 1-3 PM'
                }
            ]
        };
    }

    loadOverviewData() {
        // Update metrics
        document.getElementById('totalStudents').textContent = this.sampleData.students.length;
        document.getElementById('atRiskStudents').textContent = 
            this.sampleData.students.filter(s => s.risk === 'high').length;
        document.getElementById('successRate').textContent = '87%';
        document.getElementById('avgGPA').textContent = '3.2';

        // Update charts
        this.updateGradeChart();
        this.updateDepartmentChart();
    }

    updateGradeChart() {
        const ctx = document.getElementById('gradeChart');
        if (this.charts.gradeChart) {
            this.charts.gradeChart.destroy();
        }

        this.charts.gradeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['A', 'B', 'C', 'D', 'F'],
                datasets: [{
                    data: [25, 35, 20, 15, 5],
                    backgroundColor: ['#38a169', '#68d391', '#f6e05e', '#f6ad55', '#e53e3e'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    updateDepartmentChart() {
        const ctx = document.getElementById('departmentChart');
        if (this.charts.departmentChart) {
            this.charts.departmentChart.destroy();
        }

        this.charts.departmentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Computer Science', 'Mathematics', 'Biology', 'Engineering'],
                datasets: [{
                    label: 'Average GPA',
                    data: [3.4, 3.6, 3.2, 3.1],
                    backgroundColor: '#667eea',
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 4.0
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

    loadRiskAnalysisData() {
        const riskCounts = {
            high: this.sampleData.students.filter(s => s.risk === 'high').length,
            medium: this.sampleData.students.filter(s => s.risk === 'medium').length,
            low: this.sampleData.students.filter(s => s.risk === 'low').length
        };

        document.getElementById('highRiskCount').textContent = riskCounts.high;
        document.getElementById('mediumRiskCount').textContent = riskCounts.medium;
        document.getElementById('lowRiskCount').textContent = riskCounts.low;

        this.updateRiskStudentsTable();
    }

    updateRiskStudentsTable() {
        const tbody = document.getElementById('riskStudentsTable');
        const filteredStudents = this.filterRiskStudents();

        tbody.innerHTML = filteredStudents.map(student => {
            const riskFactors = this.getRiskFactors(student);
            const recommendations = this.getRecommendations(student.risk);

            return `
                <tr>
                    <td>${student.id}</td>
                    <td>${student.name}</td>
                    <td>${student.major}</td>
                    <td><span class="risk-badge ${student.risk}">${student.risk.toUpperCase()}</span></td>
                    <td>${student.gpa}</td>
                    <td>${riskFactors.join(', ')}</td>
                    <td>${recommendations.join(', ')}</td>
                </tr>
            `;
        }).join('');
    }

    filterRiskStudents() {
        const riskFilter = document.getElementById('riskLevelFilter')?.value || 'all';
        const departmentFilter = document.getElementById('departmentFilter')?.value || 'all';

        return this.sampleData.students.filter(student => {
            const riskMatch = riskFilter === 'all' || student.risk === riskFilter;
            const departmentMatch = departmentFilter === 'all' || 
                student.major.toLowerCase().includes(departmentFilter.toLowerCase());
            return riskMatch && departmentMatch;
        });
    }

    getRiskFactors(student) {
        const factors = [];
        if (student.gpa < 3.0) factors.push('Low GPA');
        if (student.gpa < 2.5) factors.push('Academic Probation');
        if (student.risk === 'high') factors.push('Multiple Failed Courses');
        if (student.risk === 'medium') factors.push('Declining Performance');
        return factors.length > 0 ? factors : ['No Major Risk Factors'];
    }

    getRecommendations(riskLevel) {
        switch(riskLevel) {
            case 'high':
                return ['Immediate Advisor Meeting', 'Academic Support Services', 'Tutoring'];
            case 'medium':
                return ['Monitor Closely', 'Study Skills Workshop', 'Peer Mentoring'];
            case 'low':
                return ['Maintain Current Support', 'Optional Enrichment'];
            default:
                return ['Continue Current Path'];
        }
    }

    generateOptimalCoursePath() {
        const major = document.getElementById('majorSelect')?.value || 'computer-science';
        
        // Update path metrics
        document.getElementById('totalCredits').textContent = '120';
        document.getElementById('estimatedDuration').textContent = '4 years';
        document.getElementById('successProbability').textContent = '87%';

        // Create course path visualization
        this.createCoursePathVisualization(major);
    }

    createCoursePathVisualization(major) {
        const diagram = document.getElementById('coursePathDiagram');
        
        // Sample course sequences based on major
        const paths = {
            'computer-science': [
                { semester: 'Fall Year 1', courses: ['CSEE 200', 'MATH 151', 'ENGL 100'] },
                { semester: 'Spring Year 1', courses: ['CSEE 201', 'MATH 152', 'PHYS 121'] },
                { semester: 'Fall Year 2', courses: ['CSEE 301', 'MATH 251', 'CSEE 300'] },
                { semester: 'Spring Year 2', courses: ['CSEE 302', 'CSEE 400', 'MATH 301'] }
            ],
            'mathematics': [
                { semester: 'Fall Year 1', courses: ['MATH 151', 'PHYS 121', 'ENGL 100'] },
                { semester: 'Spring Year 1', courses: ['MATH 152', 'MATH 221', 'CSEE 200'] },
                { semester: 'Fall Year 2', courses: ['MATH 251', 'MATH 301', 'MATH 221'] },
                { semester: 'Spring Year 2', courses: ['MATH 302', 'MATH 401', 'STAT 355'] }
            ]
        };

        const selectedPath = paths[major] || paths['computer-science'];
        
        diagram.innerHTML = `
            <div class="path-timeline">
                ${selectedPath.map(semester => `
                    <div class="semester-block">
                        <h4>${semester.semester}</h4>
                        <div class="courses">
                            ${semester.courses.map(course => `
                                <div class="course-item">${course}</div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        // Add some basic styling
        const style = document.createElement('style');
        style.textContent = `
            .path-timeline {
                display: flex;
                flex-direction: column;
                gap: 20px;
                width: 100%;
            }
            .semester-block {
                background: #f7fafc;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .semester-block h4 {
                margin: 0 0 10px 0;
                color: #2d3748;
            }
            .courses {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            .course-item {
                background: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 0.9rem;
                color: #4a5568;
                border: 1px solid #e2e8f0;
            }
        `;
        document.head.appendChild(style);
    }

    generateStudyGroups() {
        // This would typically make an API call to generate study groups
        // For now, we'll use the sample data
        console.log('Generating study groups...');
        
        // Update the study groups display with sample data
        const groupsGrid = document.querySelector('.study-groups-grid');
        if (groupsGrid) {
            groupsGrid.innerHTML = this.sampleData.studyGroups.map(group => `
                <div class="study-group-card">
                    <h4>${group.name} - ${group.course}</h4>
                    <div class="group-members">
                        ${group.members.map(member => `<div class="member">${member}</div>`).join('')}
                    </div>
                    <div class="group-metrics">
                        <div class="metric">Compatibility: ${group.compatibility}%</div>
                        <div class="metric">Avg GPA: ${group.avgGpa}</div>
                        <div class="metric">Meeting Time: ${group.meetingTime}</div>
                    </div>
                    <div class="group-actions">
                        <button class="action-btn primary">Schedule Meeting</button>
                        <button class="action-btn secondary">View Details</button>
                    </div>
                </div>
            `).join('');
        }
    }

    loadPredictionData() {
        // Update prediction insights and charts
        this.updateSuccessProbabilityChart();
        this.updateRiskFactorChart();
    }

    updateSuccessProbabilityChart() {
        const ctx = document.getElementById('successProbabilityChart');
        if (this.charts.successProbabilityChart) {
            this.charts.successProbabilityChart.destroy();
        }

        this.charts.successProbabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['CSEE 200', 'MATH 151', 'BIOL 141', 'CSEE 201', 'MATH 152'],
                datasets: [{
                    label: 'Success Probability (%)',
                    data: [89, 76, 82, 85, 71],
                    backgroundColor: ['#38a169', '#e53e3e', '#d69e2e', '#38a169', '#e53e3e'],
                    borderRadius: 8
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
    }

    updateRiskFactorChart() {
        const ctx = document.getElementById('riskFactorChart');
        if (this.charts.riskFactorChart) {
            this.charts.riskFactorChart.destroy();
        }

        this.charts.riskFactorChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low GPA', 'Missing Prerequisites', 'Course Load', 'Attendance', 'Other'],
                datasets: [{
                    data: [35, 25, 20, 15, 5],
                    backgroundColor: ['#e53e3e', '#d69e2e', '#f6e05e', '#68d391', '#a0aec0'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    refreshPredictions() {
        console.log('Refreshing predictions...');
        // This would typically make an API call to refresh predictions
        this.loadPredictionData();
    }

    exportPredictions() {
        console.log('Exporting predictions...');
        // This would typically export data to CSV or PDF
        alert('Export functionality would be implemented here');
    }

    initializeCharts() {
        // Initialize any charts that need to be created immediately
        console.log('Dashboard initialized');
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AdvisorDashboard();
});
