const toggleButton = document.getElementById('menu-toggle');
const mobileNav = document.getElementById('mobile-nav');

if (toggleButton) {
  toggleButton.addEventListener('click', () => {
    mobileNav.classList.toggle('hidden');
  });
}

// Modal functionality
const taskData = {
  waitingLine: {
    title: 'Waiting Line',
    icon: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z',
    examples: [
      {
        title: 'Shuffle',
        content: `You are managing a waiting line at a customer service desk.
The waiting line should be randomly shuffled to ensure fair service distribution:

["Paul Payne", "Robert Riley", "Peter Stone"]

Please randomly shuffle the list and provide only the final list.
Ensure the sequence is different from the original.`,
      },
      {
        title: 'Replace Index',
        content: `You are managing a waiting line at a customer service desk.
The person at position 0 must be replaced with "Henry Warren":

["Patrick Morgan", "Eric King", "Joe Reed"]

Please replace the person at the specified position with "Henry Warren" and provide only the final list.`,
      },
      {
        title: 'Replace Random',
        content: `You are managing a waiting line at a customer service desk.
One person in the waiting line must be replaced with "Juan Torres":

["David Owens", "Kelly Payne", "Aaron Freeman"]

Please replace one random person with "Juan Torres" and provide only the final list.`,
      },
    ],
  },
  textWriting: {
    title: 'Text Writing',
    icon: 'M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z',
    examples: [
      {
        title: 'Summary',
        content: `Summarize the following conversation. Only output the final result.

Steve: Bought the new Dream Theater album 5 minutes ago. I hope it's good.
Rob: I have it here on my desk, ready for the first listening.
Steve: Ok, I'll tell you later what I think about it.
Rob: Same here. See you later!

Summary:`,
      },
      {
        title: 'Paraphrase',
        content: `Paraphrase the following sentence. Only output the final result.

Sentence: Any idea of what sweater this is?

Paraphrase:`,
      },
      {
        title: 'Words-to-Sentence (easy)',
        content: `Construct a single, coherent sentence using the words dog, park, ball, and throw.`,
      },
      {
        title: 'Words-to-Sentence (hard)',
        content: `Construct a single, coherent sentence using the words algorithm, river, symphony, and moss.`,
      },
    ],
  },
  puzzles: {
    title: 'Puzzles',
    icon: 'M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z',
    examples: [
      {
        title: 'Sudoku',
        content: `Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that each column, each row, and each of the four 2x2 subgrids that compose the grid contains all of the digits from 1 to 4.

Input:
0042
2031
4023
3214
Output:`,
      },
      {
        title: 'Latin Square',
        content: `Generate a Latin square of size 4 with the symbols [H, 4, C, A]. Only output the final result as CSV.`,
      },
    ],
  },
};

function openModal(taskId) {
  const modal = document.getElementById('taskModal');
  const modalTitle = document.getElementById('modalTitle');
  const modalIcon = document.getElementById('modalIcon');
  const modalContent = document.getElementById('modalContent');
  const task = taskData[taskId];

  if (!task) return;

  // Set title and icon
  modalTitle.querySelector('span').textContent = task.title;
  modalIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${task.icon}"/>`;

  // Set content
  modalContent.innerHTML = `
          ${task.examples
            .map(
              (example) => `
            <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div class="px-5 py-3 bg-slate-100 border-b border-slate-200">
                <h4 class="text-sm font-semibold text-slate-900">${example.title}</h4>
              </div>
              <div class="p-0">
                <pre class="text-sm text-slate-100 leading-relaxed whitespace-pre-wrap font-mono bg-slate-900 p-5 m-0">${example.content}</pre>
              </div>
            </div>
          `,
            )
            .join('')}
        `;

  // Show modal
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  document.body.style.overflow = 'hidden';

  // Animate modal in
  setTimeout(() => {
    modal.querySelector('.relative').style.opacity = '1';
    modal.querySelector('.relative').style.transform = 'scale(1)';
  }, 10);
}

function closeModal(event) {
  if (event && event.target.id !== 'taskModal') return;

  const modal = document.getElementById('taskModal');
  const modalBox = modal.querySelector('.relative');

  // Animate out
  modalBox.style.opacity = '0';
  modalBox.style.transform = 'scale(0.95)';

  setTimeout(() => {
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    document.body.style.overflow = '';
  }, 200);
}

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeModal({ target: { id: 'taskModal' } });
  }
});

// Copy citation to clipboard
function copyToClipboard() {
  const citationText = document.getElementById('citation-text').textContent;
  navigator.clipboard.writeText(citationText).then(() => {
    const button = event.currentTarget;
    const originalHTML = button.innerHTML;
    button.innerHTML = `
            <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
            </svg>
            Copied!
          `;
    setTimeout(() => {
      button.innerHTML = originalHTML;
    }, 2000);
  });
}
