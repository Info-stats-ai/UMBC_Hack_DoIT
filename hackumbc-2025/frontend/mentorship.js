// Mentorship Page JavaScript

let allMentors = [];
let currentFilter = 'all';

// DOM Elements
const mentorshipForm = document.getElementById('mentorship-form');
const resultsSection = document.getElementById('results-section');
const mentorsGrid = document.getElementById('mentors-grid');
const loadingElement = document.getElementById('loading');
const errorElement = document.getElementById('error-message');
const errorText = document.getElementById('error-text');
const matchesCount = document.getElementById('matches-count');
const statsGrid = document.getElementById('stats-grid');

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    loadMentorStatistics();
    setupEventListeners();
    
    // Get student ID and load details
    const studentId = getStudentId();
    if (studentId) {
        loadStudentDetails(studentId);
    }
});

function getStudentId() {
    // First check URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    let studentId = urlParams.get('student_id');
    
    // If not in URL, check session storage
    if (!studentId) {
        studentId = sessionStorage.getItem('studentId');
    }
    
    // If not in session, check local storage
    if (!studentId) {
        studentId = localStorage.getItem('studentId');
    }
    
    return studentId;
}

async function loadStudentDetails(studentId) {
    try {
        const response = await fetch(`/student/${studentId}`);
        if (response.ok) {
            const studentData = await response.json();
            
            // Pre-populate the form with student data
            populateStudentForm(studentData);
        } else {
            console.warn('Student not found:', studentId);
            // Still set the student ID even if details not found
            const studentIdField = document.getElementById('student-id');
            if (studentIdField) {
                studentIdField.value = studentId;
            }
        }
    } catch (error) {
        console.error('Error loading student details:', error);
        // Still set the student ID even if there's an error
        const studentIdField = document.getElementById('student-id');
        if (studentIdField) {
            studentIdField.value = studentId;
        }
    }
}

function populateStudentForm(studentData) {
    // Populate form fields with student data
    const fields = {
        'student-id': studentData.student_id,
        'name': studentData.name,
        'major': studentData.major || 'Computer Science', // Default if not available
        'year-level': getYearLevelValue(studentData.year_level),
        'availability': getAvailabilityValue(studentData.work_hours)
    };
    
    // Set form field values
    Object.entries(fields).forEach(([fieldId, value]) => {
        const field = document.getElementById(fieldId);
        if (field && value) {
            field.value = value;
        }
    });
    
    // Populate academic interests based on major
    const interestsField = document.getElementById('interests');
    if (interestsField && studentData.major) {
        const defaultInterests = getDefaultInterests(studentData.major);
        interestsField.value = defaultInterests;
    }
}

function getYearLevelValue(yearLevel) {
    if (!yearLevel) return 'Freshman (1st Year)';
    
    const yearMap = {
        'freshman': 'Freshman (1st Year)',
        'sophomore': 'Sophomore (2nd Year)', 
        'junior': 'Junior (3rd Year)',
        'senior': 'Senior (4th Year)',
        '1': 'Freshman (1st Year)',
        '2': 'Sophomore (2nd Year)',
        '3': 'Junior (3rd Year)',
        '4': 'Senior (4th Year)'
    };
    
    const normalized = yearLevel.toString().toLowerCase();
    return yearMap[normalized] || 'Freshman (1st Year)';
}

function getAvailabilityValue(workHours) {
    if (!workHours) return 'Flexible';
    
    if (workHours === 0) return 'Flexible';
    if (workHours <= 20) return 'Part-time';
    return 'Limited';
}

function getDefaultInterests(major) {
    const interestMap = {
        'Computer Science': 'Machine Learning, Data Science, Software Engineering',
        'Biology': 'Molecular Biology, Genetics, Research',
        'Mathematics': 'Statistics, Applied Mathematics, Data Analysis',
        'Physics': 'Theoretical Physics, Research, Engineering',
        'Chemistry': 'Organic Chemistry, Research, Laboratory Work'
    };
    
    return interestMap[major] || 'Academic Excellence, Research, Career Development';
}

function setupEventListeners() {
    // Form submission
    if (mentorshipForm) {
        mentorshipForm.addEventListener('submit', handleFormSubmit);
    }

    // Filter tabs
    const filterTabs = document.querySelectorAll('.filter-tab');
    filterTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const filter = this.dataset.filter;
            setActiveFilter(filter);
            filterMentors(filter);
        });
    });
}

function handleFormSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(mentorshipForm);
    const mentorshipRequest = {
        mentee_id: formData.get('mentee_id'),
        name: formData.get('name') || 'Student',
        major: formData.get('major'),
        year_level: parseInt(formData.get('year_level')),
        academic_interests: formData.get('academic_interests') ? 
            formData.get('academic_interests').split(',').map(s => s.trim()) : [],
        career_goals: formData.get('career_goals') ? 
            formData.get('career_goals').split(',').map(s => s.trim()) : [],
        skills_to_develop: formData.get('skills_to_develop') ? 
            formData.get('skills_to_develop').split(',').map(s => s.trim()) : [],
        preferred_mentor_type: formData.getAll('preferred_mentor_type'),
        availability: formData.get('availability'),
        goals: formData.get('goals') || 'Academic and career guidance'
    };

    // Validate required fields
    if (!mentorshipRequest.mentee_id) {
        showError('Please enter your Student ID');
        return;
    }

    if (mentorshipRequest.preferred_mentor_type.length === 0) {
        showError('Please select at least one mentor type');
        return;
    }

    findMentors(mentorshipRequest);
}

async function findMentors(request) {
    showLoading();
    hideError();
    
    try {
        const response = await fetch('/mentorship/find-mentors', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const mentors = await response.json();
        allMentors = mentors;
        
        hideLoading();
        displayMentors(mentors);
        showResults();
        
    } catch (error) {
        console.error('Error finding mentors:', error);
        hideLoading();
        showError('Failed to find mentors. Please try again.');
    }
}

function displayMentors(mentors) {
    if (!mentors || mentors.length === 0) {
        mentorsGrid.innerHTML = `
            <div class="no-results">
                <i class="fas fa-search"></i>
                <h3>No mentors found</h3>
                <p>Try adjusting your criteria or selecting different mentor types.</p>
            </div>
        `;
        matchesCount.textContent = '0';
        return;
    }

    matchesCount.textContent = mentors.length;
    
    mentorsGrid.innerHTML = mentors.map(match => createMentorCard(match)).join('');
}

function createMentorCard(match) {
    const mentor = match.mentor;
    const initials = mentor.name.split(' ').map(n => n[0]).join('');
    const compatibilityPercent = Math.round(match.compatibility_score * 100);
    
    // Generate stars for rating
    const stars = generateStars(mentor.rating);
    
    // Format specializations
    const specializations = mentor.specializations.slice(0, 3).map(spec => 
        `<span class="specialization-tag">${spec}</span>`
    ).join('');
    
    // Determine mentor type display
    const mentorTypeDisplay = {
        'senior_student': 'Senior Student',
        'faculty': 'Faculty'
    };

    return `
        <div class="mentor-card" data-mentor-type="${mentor.mentor_type}">
            <div class="mentor-type-badge ${mentor.mentor_type}">
                ${mentorTypeDisplay[mentor.mentor_type]}
            </div>
            
            <div class="mentor-header">
                <div class="mentor-avatar">${initials}</div>
                <div class="mentor-info">
                    <h3>${mentor.name}</h3>
                    <div class="mentor-department">${mentor.department}</div>
                    <div class="mentor-rating">
                        <span class="stars">${stars}</span>
                        <span>${mentor.rating}</span>
                    </div>
                </div>
            </div>

            <div class="compatibility-score">
                ${compatibilityPercent}% Compatibility Match
            </div>

            <div class="mentor-details">
                <div class="detail-row">
                    <span class="detail-label">Experience:</span>
                    <span class="detail-value">${mentor.year_level} ${mentor.mentor_type === 'senior_student' ? 'year' : 'years'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Availability:</span>
                    <span class="detail-value">${mentor.availability}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Capacity:</span>
                    <span class="detail-value">${mentor.current_mentees}/${mentor.mentoring_capacity}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Meeting Frequency:</span>
                    <span class="detail-value">${match.recommended_meeting_frequency}</span>
                </div>
            </div>

            <div class="mentor-specializations">
                <div class="detail-label">Specializations:</div>
                <div class="specializations-list">
                    ${specializations}
                </div>
            </div>

            <div class="mentor-actions">
                <button class="btn-outline btn-small" onclick="viewMentorDetails('${match.match_id}')">
                    <i class="fas fa-info-circle"></i> View Details
                </button>
                <button class="btn-primary btn-small" onclick="connectWithMentor('${mentor.mentor_id}', '${mentor.contact_info}')">
                    <i class="fas fa-envelope"></i> Connect
                </button>
            </div>
        </div>
    `;
}

function generateStars(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    let stars = '';
    
    for (let i = 0; i < fullStars; i++) {
        stars += '★';
    }
    
    if (hasHalfStar) {
        stars += '☆';
    }
    
    // Fill remaining with empty stars up to 5
    const remainingStars = 5 - Math.ceil(rating);
    for (let i = 0; i < remainingStars; i++) {
        stars += '☆';
    }
    
    return stars;
}

function setActiveFilter(filter) {
    currentFilter = filter;
    
    // Update active tab
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
}

function filterMentors(filter) {
    const mentorCards = document.querySelectorAll('.mentor-card');
    let visibleCount = 0;
    
    mentorCards.forEach(card => {
        const mentorType = card.dataset.mentorType;
        
        if (filter === 'all' || mentorType === filter) {
            card.classList.remove('hidden');
            visibleCount++;
        } else {
            card.classList.add('hidden');
        }
    });
    
    matchesCount.textContent = visibleCount;
}

function viewMentorDetails(matchId) {
    const match = allMentors.find(m => m.match_id === matchId);
    if (!match) return;
    
    const mentor = match.mentor;
    const modalContent = document.getElementById('mentor-details');
    
    // Format compatibility factors
    const compatibilityFactors = Object.entries(match.compatibility_factors)
        .map(([factor, score]) => {
            const percentage = Math.round(score * 100);
            const label = factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            return `
                <div class="compatibility-factor">
                    <div class="factor-label">${label}</div>
                    <div class="factor-bar">
                        <div class="factor-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div class="factor-score">${percentage}%</div>
                </div>
            `;
        }).join('');
    
    modalContent.innerHTML = `
        <div class="mentor-detail-header">
            <div class="mentor-avatar-large">${mentor.name.split(' ').map(n => n[0]).join('')}</div>
            <div class="mentor-detail-info">
                <h3>${mentor.name}</h3>
                <p>${mentor.department} • ${mentor.mentor_type.replace('_', ' ')}</p>
                <div class="mentor-rating">
                    <span class="stars">${generateStars(mentor.rating)}</span>
                    <span>${mentor.rating}/5.0</span>
                </div>
            </div>
        </div>

        <div class="mentor-bio">
            <h4>About</h4>
            <p>${mentor.bio}</p>
        </div>

        <div class="compatibility-analysis">
            <h4>Compatibility Analysis (${Math.round(match.compatibility_score * 100)}% match)</h4>
            <div class="compatibility-factors">
                ${compatibilityFactors}
            </div>
        </div>

        <div class="mentor-details-grid">
            <div class="detail-section">
                <h4>Research Interests</h4>
                <div class="tags-list">
                    ${mentor.research_interests.map(interest => `<span class="tag">${interest}</span>`).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h4>Skills</h4>
                <div class="tags-list">
                    ${mentor.skills.map(skill => `<span class="tag">${skill}</span>`).join('')}
                </div>
            </div>


            <div class="detail-section">
                <h4>Suggested Activities</h4>
                <ul>
                    ${match.suggested_activities.map(activity => `<li>${activity}</li>`).join('')}
                </ul>
            </div>

            <div class="detail-section">
                <h4>Mentorship Goals</h4>
                <ul>
                    ${match.goals.map(goal => `<li>${goal}</li>`).join('')}
                </ul>
            </div>
        </div>

        <div class="contact-info">
            <h4>Contact Information</h4>
            <p><i class="fas fa-envelope"></i> ${mentor.contact_info}</p>
            <p><i class="fas fa-clock"></i> Available: ${mentor.availability}</p>
            <p><i class="fas fa-calendar"></i> Recommended: ${match.recommended_meeting_frequency}</p>
        </div>
    `;
    
    // Store current mentor for connect action
    window.currentMentorDetails = {
        mentor_id: mentor.mentor_id,
        contact_info: mentor.contact_info,
        name: mentor.name
    };
    
    document.getElementById('mentor-modal').style.display = 'block';
}

function connectWithMentor(mentorId, contactInfo) {
    if (contactInfo) {
        window.open(`mailto:${contactInfo}?subject=Mentorship Request - UMBC Academic Success Platform&body=Hi,%0D%0A%0D%0AI found your profile through the UMBC Academic Success Platform's mentorship program and would like to connect for mentorship guidance.%0D%0A%0D%0AThank you!`, '_blank');
    } else {
        showError('Contact information not available for this mentor.');
    }
}

function closeMentorModal() {
    document.getElementById('mentor-modal').style.display = 'none';
}

// Connect button in modal
window.connectWithMentor = function() {
    if (window.currentMentorDetails) {
        connectWithMentor(window.currentMentorDetails.mentor_id, window.currentMentorDetails.contact_info);
        closeMentorModal();
    }
};

async function loadMentorStatistics() {
    try {
        const response = await fetch('/mentorship/statistics');
        if (!response.ok) throw new Error('Failed to load statistics');
        
        const stats = await response.json();
        displayStatistics(stats);
        
    } catch (error) {
        console.error('Error loading statistics:', error);
        // Show default stats or hide section
        statsGrid.innerHTML = `
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-users"></i></div>
                <div class="stat-number">-</div>
                <div class="stat-label">Loading...</div>
            </div>
        `;
    }
}

function displayStatistics(stats) {
    const statsHTML = `
        <div class="stat-card">
            <div class="stat-icon"><i class="fas fa-users"></i></div>
            <div class="stat-number">${stats.total_mentors}</div>
            <div class="stat-label">Total Mentors</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-icon"><i class="fas fa-user-graduate"></i></div>
            <div class="stat-number">${stats.senior_student_mentors}</div>
            <div class="stat-label">Senior Students</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-icon"><i class="fas fa-chalkboard-teacher"></i></div>
            <div class="stat-number">${stats.faculty_mentors}</div>
            <div class="stat-label">Faculty Members</div>
        </div>
        
        
        <div class="stat-card">
            <div class="stat-icon"><i class="fas fa-star"></i></div>
            <div class="stat-number">${stats.average_rating}</div>
            <div class="stat-label">Avg Rating</div>
        </div>
    `;
    
    statsGrid.innerHTML = statsHTML;
}

function showLoading() {
    loadingElement.style.display = 'block';
    resultsSection.style.display = 'none';
    errorElement.style.display = 'none';
}

function hideLoading() {
    loadingElement.style.display = 'none';
}

function showResults() {
    resultsSection.style.display = 'block';
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    errorText.textContent = message;
    errorElement.style.display = 'block';
    resultsSection.style.display = 'none';
    loadingElement.style.display = 'none';
}

function hideError() {
    errorElement.style.display = 'none';
}

// Modal event listeners
window.onclick = function(event) {
    const modal = document.getElementById('mentor-modal');
    if (event.target === modal) {
        closeMentorModal();
    }
};

// Add styles for modal content that's generated dynamically
const additionalStyles = `
<style>
.mentor-detail-header {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 25px;
    padding-bottom: 20px;
    border-bottom: 1px solid #eee;
}

.mentor-avatar-large {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 2rem;
}

.mentor-detail-info h3 {
    margin: 0 0 5px 0;
    font-size: 1.5rem;
    color: #333;
}

.mentor-detail-info p {
    margin: 0 0 10px 0;
    color: #666;
    text-transform: capitalize;
}

.mentor-bio {
    margin-bottom: 25px;
}

.mentor-bio h4,
.compatibility-analysis h4,
.detail-section h4,
.contact-info h4 {
    color: #333;
    margin-bottom: 15px;
    font-size: 1.1rem;
}

.compatibility-analysis {
    margin-bottom: 25px;
}

.compatibility-factors {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.compatibility-factor {
    display: flex;
    align-items: center;
    gap: 15px;
}

.factor-label {
    min-width: 150px;
    font-weight: 500;
    color: #555;
    font-size: 0.9rem;
}

.factor-bar {
    flex: 1;
    height: 8px;
    background: #e0e8ff;
    border-radius: 4px;
    overflow: hidden;
}

.factor-fill {
    height: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    transition: width 0.3s ease;
}

.factor-score {
    min-width: 40px;
    font-weight: 600;
    color: #667eea;
    font-size: 0.9rem;
}

.mentor-details-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
    margin-bottom: 25px;
}

.detail-section {
    background: #f8f9ff;
    padding: 20px;
    border-radius: 10px;
}

.tags-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.tag {
    background: #667eea;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

.detail-section ul {
    list-style: none;
    padding: 0;
}

.detail-section li {
    padding: 5px 0;
    color: #555;
    position: relative;
    padding-left: 20px;
}

.detail-section li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: #667eea;
    font-weight: bold;
}

.contact-info {
    background: #f8f9ff;
    padding: 20px;
    border-radius: 10px;
}

.contact-info p {
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #555;
}

.contact-info i {
    color: #667eea;
    width: 16px;
}

.no-results {
    grid-column: 1 / -1;
    text-align: center;
    padding: 80px 20px;
    color: #666;
}

.no-results i {
    font-size: 3rem;
    margin-bottom: 20px;
    color: #ccc;
}

.no-results h3 {
    margin-bottom: 10px;
    color: #333;
}

@media (max-width: 768px) {
    .mentor-details-grid {
        grid-template-columns: 1fr;
    }
    
    .mentor-detail-header {
        flex-direction: column;
        text-align: center;
    }
    
    .compatibility-factor {
        flex-direction: column;
        align-items: stretch;
        gap: 8px;
    }
    
    .factor-label {
        min-width: auto;
    }
    
    .factor-score {
        text-align: center;
    }
}
</style>
`;

// Add the additional styles to the document head
document.head.insertAdjacentHTML('beforeend', additionalStyles);
