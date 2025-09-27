# 📊 Titanic EDA - Data Dictionary

## 📋 Dataset Overview
- **Source**: Kaggle Titanic: Machine Learning from Disaster
- **URL**: https://www.kaggle.com/c/titanic/data
- **Records**: 891 passengers
- **Features**: 18 total (12 original + 6 engineered)
- **Target Variable**: Survived

## 🔤 Original Features

### Passenger Identification
| Feature | Type | Description | Example Values | Missing |
|---------|------|-------------|----------------|---------|
| **PassengerId** | int64 | Unique passenger ID | 1, 2, 3, ... 891 | 0 |
| **Name** | object | Full passenger name | "Braund, Mr. Owen Harris" | 0 |
| **Ticket** | object | Ticket number | "A/5 21171", "PC 17599" | 0 |

### Demographics
| Feature | Type | Description | Example Values | Missing |
|---------|------|-------------|----------------|---------|
| **Sex** | object | Passenger gender | "male", "female" | 0 |
| **Age** | float64 | Age in years | 22.0, 35.0, 54.0 | 177 (19.9%) |

### Travel Details
| Feature | Type | Description | Example Values | Missing |
|---------|------|-------------|----------------|---------|
| **Pclass** | int64 | Ticket class | 1 (1st), 2 (2nd), 3 (3rd) | 0 |
| **Fare** | float64 | Passenger fare | 7.25, 71.28, 512.33 | 0 |
| **Cabin** | object | Cabin number | "C85", "B96 B98", null | 687 (77.1%) |
| **Embarked** | object | Port of embarkation | "C", "Q", "S" | 2 (0.2%) |

### Family Information
| Feature | Type | Description | Example Values | Missing |
|---------|------|-------------|----------------|---------|
| **SibSp** | int64 | Siblings/spouses aboard | 0, 1, 2, 3, 4, 5, 8 | 0 |
| **Parch** | int64 | Parents/children aboard | 0, 1, 2, 3, 4, 5, 6 | 0 |

### Target Variable
| Feature | Type | Description | Example Values | Missing |
|---------|------|-------------|----------------|---------|
| **Survived** | int64 | Survival status | 0 (No), 1 (Yes) | 0 |

## 🔧 Engineered Features

### Family Composition
| Feature | Type | Description | Formula | Values |
|---------|------|-------------|---------|---------|
| **FamilySize** | int64 | Total family members | SibSp + Parch + 1 | 1-11 |
| **IsAlone** | int64 | Traveling alone flag | 1 if FamilySize == 1, else 0 | 0, 1 |

### Categorical Derived
| Feature | Type | Description | Categories | Distribution |
|---------|------|-------------|------------|-------------|
| **Title** | object | Extracted from Name | Mr, Mrs, Miss, Master, Rare | Mr: 517, Miss: 182, Mrs: 125 |
| **AgeGroup** | category | Age categories | Child, Teen, Adult, Middle Age, Senior | Adult: 398, Middle Age: 220 |
| **FareGroup** | category | Fare quartiles | Low, Medium, High, Very High | Equal ~223 each |

### Binary Indicators
| Feature | Type | Description | Logic | Values |
|---------|------|-------------|-------|---------|
| **HasCabin** | int64 | Cabin info available | 1 if Cabin not null, else 0 | 0: 687, 1: 204 |

## 📊 Detailed Value Distributions

### Passenger Class (Pclass)
```
1st Class: 216 passengers (24.2%)
2nd Class: 184 passengers (20.7%)  
3rd Class: 491 passengers (55.1%)
```

### Gender (Sex)
```
Male:   577 passengers (64.8%)
Female: 314 passengers (35.2%)
```

### Age Groups (AgeGroup)
```
Child (0-12):      68 passengers (7.6%)
Teen (13-18):      41 passengers (4.6%)
Adult (19-35):    398 passengers (44.7%)
Middle Age (36-60): 220 passengers (24.7%)
Senior (60+):      44 passengers (4.9%)
Missing:          120 passengers (13.5%)
```

### Embarkation (Embarked)
```
Southampton (S): 644 passengers (72.3%)
Cherbourg (C):   168 passengers (18.9%)
Queenstown (Q):   77 passengers (8.6%)
Missing:           2 passengers (0.2%)
```

### Family Size (FamilySize)
```
1 (Alone):    537 passengers (60.3%)
2:           161 passengers (18.1%)
3:           102 passengers (11.4%)
4:            29 passengers (3.3%)
5:            15 passengers (1.7%)
6:            12 passengers (1.3%)
7:             6 passengers (0.7%)
8:             6 passengers (0.7%)
11:            4 passengers (0.4%)
```

### Titles (Title)
```
Mr:      517 passengers (58.0%)
Miss:    182 passengers (20.4%)
Mrs:     125 passengers (14.0%)
Master:   40 passengers (4.5%)
Rare:     27 passengers (3.0%)
```

## 🔍 Data Quality Metrics

### Completeness by Feature
| Feature | Complete | Missing | Completeness |
|---------|----------|---------|--------------|
| PassengerId | 891 | 0 | 100.0% |
| Survived | 891 | 0 | 100.0% |
| Pclass | 891 | 0 | 100.0% |
| Name | 891 | 0 | 100.0% |
| Sex | 891 | 0 | 100.0% |
| **Age** | 714 | 177 | **80.1%** |
| SibSp | 891 | 0 | 100.0% |
| Parch | 891 | 0 | 100.0% |
| Ticket | 891 | 0 | 100.0% |
| Fare | 891 | 0 | 100.0% |
| **Cabin** | 204 | 687 | **22.9%** |
| **Embarked** | 889 | 2 | **99.8%** |

### Statistical Summary - Numerical Features
| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| Age | 29.70 | 14.53 | 0.42 | 20.12 | 28.00 | 38.00 | 80.00 |
| SibSp | 0.52 | 1.10 | 0 | 0 | 0 | 1 | 8 |
| Parch | 0.38 | 0.81 | 0 | 0 | 0 | 0 | 6 |
| Fare | 32.20 | 49.69 | 0.00 | 7.91 | 14.45 | 31.00 | 512.33 |
| FamilySize | 1.90 | 1.61 | 1 | 1 | 1 | 2 | 11 |

## 🎯 Feature Engineering Notes

### Missing Value Treatment
- **Age**: Imputed using median values grouped by Pclass and Sex
- **Embarked**: Imputed using mode (Southampton - most common port)
- **Cabin**: Converted to binary indicator rather than imputation

### Rationale for New Features
1. **FamilySize**: Combines SibSp and Parch to understand family dynamics
2. **IsAlone**: Binary version focusing on solo vs group travel
3. **Title**: Captures social status and age/gender combination
4. **AgeGroup**: Makes age analysis more interpretable
5. **FareGroup**: Normalizes fare differences for comparison
6. **HasCabin**: Proxy for passenger location and possibly class

---
*Data Dictionary Version: 1.0*  
*Last Updated: September 27, 2025*  
*Total Features in Final Dataset: 18*