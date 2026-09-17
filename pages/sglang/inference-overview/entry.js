// Preserve old topic URLs while opening the first lesson directly.
const lesson = new URL(document.querySelector('a').href);
lesson.hash = location.hash;
location.replace(lesson.href);
