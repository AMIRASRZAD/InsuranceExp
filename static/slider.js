document.addEventListener('DOMContentLoaded', () => {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const valueField = document.getElementById(`${slider.id}_value`);
        const numberInput = document.getElementById(`${slider.id}_number`);
        
        // Initialize display
        numberInput.value = slider.value;
        valueField.value = slider.value;

        // Update on slider change
        slider.addEventListener('input', () => {
            numberInput.value = slider.value;
            valueField.value = slider.value;
        });

        // Update on number input change
        numberInput.addEventListener('input', () => {
            let value = parseFloat(numberInput.value);
            if (isNaN(value) || value < 1) value = 1;
            if (value > 70000) value = 70000;
            slider.value = value;
            valueField.value = value;
        });
    });
});