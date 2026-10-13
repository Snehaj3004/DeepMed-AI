document.querySelectorAll('input[type="file"]').forEach((input) => {
    input.addEventListener("change", () => {
        const label = input.closest(".upload-area")?.querySelector("[data-file-name]");

        if (!label) {
            return;
        }

        label.textContent = input.files.length
            ? input.files[0].name
            : label.dataset.defaultText || "Choose a file";
    });
});