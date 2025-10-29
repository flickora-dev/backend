
// Genre submenu toggle
const genresToggle = document.getElementById('genresToggle');
const genresSubmenu = document.getElementById('genresSubmenu');
const submenuArrow = genresToggle.querySelector('.submenu-arrow');

genresToggle.addEventListener('click', () => {
    genresSubmenu.classList.toggle('open');
    submenuArrow.classList.toggle('open');
});

// Mobile sidebar toggle (for future responsive implementation)
function toggleMobileSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('mobile-open');
}

// Search functionality
const searchInput = document.querySelector('.search-input');
searchInput.addEventListener('input', (e) => {
    const query = e.target.value;
    // TODO: Implement search functionality
    console.log('Search query:', query);
});

// Smooth scrolling for content area
const contentArea = document.querySelector('.content-area');
if (contentArea) {
    contentArea.style.scrollBehavior = 'smooth';
}