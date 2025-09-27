"""
BMI Calculator Core Logic

This module contains the BMI calculation functions and related utilities.
"""

import math
from typing import Dict, Tuple

class BMICalculator:
    """
    A class to handle BMI calculations and categorization
    """
    
    # BMI category thresholds
    BMI_CATEGORIES = {
        'Underweight': (0, 18.5),
        'Normal weight': (18.5, 25.0),
        'Overweight': (25.0, 30.0),
        'Obesity Class I': (30.0, 35.0),
        'Obesity Class II': (35.0, 40.0),
        'Obesity Class III': (40.0, float('inf'))
    }
    
    def __init__(self):
        pass
    
    def calculate_bmi(self, weight_kg: float, height_m: float) -> Dict[str, any]:
        """
        Calculate BMI and determine health category
        
        Args:
            weight_kg (float): Weight in kilograms
            height_m (float): Height in meters
            
        Returns:
            Dict: Dictionary containing BMI value, category, and additional info
        """
        if weight_kg <= 0 or height_m <= 0:
            raise ValueError("Weight and height must be positive values")
        
        # Calculate BMI
        bmi = weight_kg / (height_m ** 2)
        
        # Determine category
        category = self._get_bmi_category(bmi)
        
        # Get healthy weight range
        healthy_range = self._get_healthy_weight_range(height_m)
        
        result = {
            'bmi': round(bmi, 1),
            'category': category,
            'weight_kg': weight_kg,
            'height_m': height_m,
            'healthy_weight_range': healthy_range,
            'weight_to_lose': self._calculate_weight_change(bmi, weight_kg, height_m, 'lose'),
            'weight_to_gain': self._calculate_weight_change(bmi, weight_kg, height_m, 'gain')
        }
        
        return result
    
    def _get_bmi_category(self, bmi: float) -> str:
        """
        Determine BMI category based on value
        
        Args:
            bmi (float): BMI value
            
        Returns:
            str: BMI category
        """
        for category, (min_val, max_val) in self.BMI_CATEGORIES.items():
            if min_val <= bmi < max_val:
                return category
        return 'Unknown'
    
    def _get_healthy_weight_range(self, height_m: float) -> Tuple[float, float]:
        """
        Calculate healthy weight range for given height
        
        Args:
            height_m (float): Height in meters
            
        Returns:
            Tuple[float, float]: Min and max healthy weight in kg
        """
        min_healthy_weight = 18.5 * (height_m ** 2)
        max_healthy_weight = 24.9 * (height_m ** 2)
        
        return (round(min_healthy_weight, 1), round(max_healthy_weight, 1))
    
    def _calculate_weight_change(self, current_bmi: float, current_weight: float, 
                               height_m: float, direction: str) -> float:
        """
        Calculate weight change needed to reach healthy BMI range
        
        Args:
            current_bmi (float): Current BMI
            current_weight (float): Current weight in kg
            height_m (float): Height in meters
            direction (str): 'lose' or 'gain'
            
        Returns:
            float: Weight change needed in kg (positive for gain, negative for loss)
        """
        if direction == 'lose' and current_bmi > 25:
            target_bmi = 24.9
            target_weight = target_bmi * (height_m ** 2)
            return round(current_weight - target_weight, 1)
        elif direction == 'gain' and current_bmi < 18.5:
            target_bmi = 18.5
            target_weight = target_bmi * (height_m ** 2)
            return round(target_weight - current_weight, 1)
        else:
            return 0.0
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert between different units
        
        Args:
            value (float): Value to convert
            from_unit (str): Source unit
            to_unit (str): Target unit
            
        Returns:
            float: Converted value
        """
        conversions = {
            ('lbs', 'kg'): lambda x: x * 0.453592,
            ('kg', 'lbs'): lambda x: x / 0.453592,
            ('ft_in', 'm'): lambda x: x * 0.0254,  # x should be total inches
            ('m', 'ft_in'): lambda x: x / 0.0254,  # returns total inches
            ('cm', 'm'): lambda x: x / 100,
            ('m', 'cm'): lambda x: x * 100
        }
        
        key = (from_unit, to_unit)
        if key in conversions:
            return conversions[key](value)
        else:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")
    
    def get_bmi_percentile(self, bmi: float, age: int, gender: str) -> Dict[str, any]:
        """
        Get BMI percentile based on age and gender (simplified version)
        
        Args:
            bmi (float): BMI value
            age (int): Age in years
            gender (str): Gender ('Male', 'Female', or other)
            
        Returns:
            Dict: Percentile information
        """
        # This is a simplified implementation
        # In a real application, you would use actual percentile data
        
        if age < 18:
            return {'percentile': None, 'note': 'Percentile calculation not available for under 18'}
        
        # Rough estimates for adult percentiles
        percentile_ranges = {
            'Underweight': 5,
            'Normal weight': 50,
            'Overweight': 75,
            'Obesity Class I': 85,
            'Obesity Class II': 95,
            'Obesity Class III': 99
        }
        
        category = self._get_bmi_category(bmi)
        estimated_percentile = percentile_ranges.get(category, 50)
        
        return {
            'percentile': estimated_percentile,
            'note': 'Estimated percentile for adults',
            'interpretation': f'Your BMI is higher than approximately {estimated_percentile}% of adults'
        }

def calculate_bmi_simple(weight_kg: float, height_m: float) -> float:
    """
    Simple BMI calculation function
    
    Args:
        weight_kg (float): Weight in kilograms
        height_m (float): Height in meters
        
    Returns:
        float: BMI value
    """
    calculator = BMICalculator()
    result = calculator.calculate_bmi(weight_kg, height_m)
    return result['bmi']

if __name__ == "__main__":
    # Test the calculator
    calculator = BMICalculator()
    
    # Test case 1: Normal weight
    result1 = calculator.calculate_bmi(70, 1.75)
    print(f"Test 1 - BMI: {result1['bmi']}, Category: {result1['category']}")
    
    # Test case 2: Overweight
    result2 = calculator.calculate_bmi(85, 1.75)
    print(f"Test 2 - BMI: {result2['bmi']}, Category: {result2['category']}")
    
    # Test case 3: Underweight
    result3 = calculator.calculate_bmi(50, 1.75)
    print(f"Test 3 - BMI: {result3['bmi']}, Category: {result3['category']}")
    
    print("✅ BMI Calculator tests completed!")