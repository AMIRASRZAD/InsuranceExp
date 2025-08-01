document.addEventListener('DOMContentLoaded', () => {
    console.log('slider.js loaded');
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const valueField = document.getElementById(`${slider.id}_value`);
        const numberInput = document.getElementById(`${slider.id}_number`);

        // Update on slider change
        slider.addEventListener('input', () => {
            let value = parseFloat(slider.value);
            if (isNaN(value) || value < 1) value = 1;
            if (value > 30000) value = 30000;
            numberInput.value = value;
            valueField.value = value;
            numberInput.setCustomValidity('');
            console.log(`Slider updated: ${numberInput.name}=${value}`);
        });

        // Update on number input change
        numberInput.addEventListener('input', () => {
            let value = parseFloat(numberInput.value);
            if (isNaN(value)) {
                numberInput.setCustomValidity('Please fill out this field');
                console.log(`Number input empty: ${numberInput.name}`);
                return;
            }
            if (value < 1) value = 1;
            if (value > 30000) value = 30000;
            slider.value = value;
            valueField.value = value;
            numberInput.setCustomValidity('');
            console.log(`Number input updated: ${numberInput.name}=${value}`);
        });

        // Ensure valid submission
        const form = slider.closest('form');
        if (form) {
            form.addEventListener('submit', (event) => {
                let value = parseFloat(numberInput.value);
                if (!numberInput.value || isNaN(value) || value < 1 || value > 30000) {
                    event.preventDefault();
                    numberInput.setCustomValidity('Please enter a value between 1 and 30,000');
                    numberInput.reportValidity();
                    console.log(`Form submission blocked: ${numberInput.name}=${numberInput.value}`);
                    return;
                }
                valueField.value = value;
                console.log(`Form submission: ${numberInput.name}=${value}`);
            });
        }
    });
});