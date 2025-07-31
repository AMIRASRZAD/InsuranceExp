document.addEventListener('DOMContentLoaded', () => {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const valueField = document.getElementById(`${slider.id}_value`);
        const submitField = document.getElementById(`${slider.id}_value_submit`);
        const numberInput = document.getElementById(`${slider.id}_number`);
        
        // Initialize display with value from previous stage or 1000 if unset
        let initialValue = parseFloat(slider.value) || 1000;
        if (isNaN(initialValue) || initialValue < 1) initialValue = 1000;
        if (initialValue > 30000) initialValue = 30000;
        slider.value = initialValue;
        numberInput.value = initialValue;
        valueField.value = initialValue;
        if (submitField) submitField.value = initialValue;

        // Update on slider change
        slider.addEventListener('input', () => {
            let value = parseFloat(slider.value);
            if (isNaN(value) || value < 1) value = 1;
            if (value > 30000) value = 30000;
            numberInput.value = value;
            valueField.value = value;
            if (submitField) submitField.value = value;
            console.log(`Slider updated: value=${value}`);
        });

        // Update on number input change
        numberInput.addEventListener('input', () => {
            let value = parseFloat(numberInput.value);
            if (isNaN(value) || value < 1) value = 1;
            if (value > 30000) value = 30000;
            slider.value = value;
            valueField.value = value;
            if (submitField) submitField.value = value;
            console.log(`Number input updated: value=${value}`);
        });

        // Ensure valid submission
        const form = slider.closest('form');
        if (form) {
            form.addEventListener('submit', (event) => {
                let value = parseFloat(slider.value);
                if (isNaN(value) || value < 1 || value > 30000) {
                    event.preventDefault();
                    alert('Please select a value between 1 and 30,000 USD.');
                    return;
                }
                valueField.value = value;
                if (submitField) submitField.value = value;
                console.log(`Form submission: final_guess_value=${value}`);
            });
        }
    });
});