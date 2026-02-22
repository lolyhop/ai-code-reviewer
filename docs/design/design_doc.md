# Automated Pull Request Reviewer: ML System Design

This document outlines the machine learning system design for Automated Pull Request Reviewer. It serves as our Contract of Work and ensures alignment across DS, ML, and Business roles.

## 1. Problem Definition

### 1.1 Context

Senior developers spend up to 20-30% of their time on code reviews. A significant portion of this effort is wasted on identifying repetitive, low-level issues that static analysis tools miss, but which are blocking for production (e.g., specific logic errors, security vulnerabilities in context, bad code practices).

Existing AI solutions (Copilot/ChatGPT) require sending proprietary code to external APIs, which violates data privacy policies in many enterprise environments.

To address these challenges, we introduce Automated Pull Request Reviewer (APR), an automated code review agent designed to act as a privacy-first quality gate. By leveraging machine learning models, our agent identifies blocking issues locally within the CI/CD pipeline. This solution effectively eliminates data leakage risks while significantly reducing the cognitive load on human reviewers by filtering out critical errors before they reach the senior team.

### 1.2 Functional Requirements

TODO:
- Хотим ли мы для каждого проекта хранить фичу его собственного стиля кода?
- Хотим ли мы классифицировать проблему, которая есть в diff hunk'е? (security, code style, code logic)
- Нужно ли нам возвращать комментарий по поводу того, что не так с кодом (так делают cloud-based аналоги)

Пока что текущий контракт:
```json
{
    "lines": [(100, 115), ...],
    "comments": ["Your code is bad.", ...]
}
```

### 1.3 Key Stakeholders & Business Impact

Our solution delivers value to different roles within the organization:

**Junior/Middle Developers**
- **Accelerated Time-to-Merge**: Receive immediate feedback on blocking issues without waiting hours for a human review;
- **Learning & Compliance**: Quickly learn team standards and best practices through consistent, automated explanations, reducing the fear of submitting "bad code" to seniors.

**Senior Developers / Tech Leads**
- **Reduced Cognitive Load**: Automate the mundane task of catching repetitive, low-level issues (e.g., style violations, obvious bugs);
- **High-Value Allocation**: Shift focus from "nitpicking" to architectural design, complex logic, and mentoring, increasing overall team velocity.

**Security / Compliance Officers**
- **Zero Data Leakage**: Run ML inference locally (On-Premise), ensuring proprietary code never leaves the secure perimeter (unlike cloud-based LLMs);
- **Consistent Scanning**: Enforce critical security checks on every single Pull Request, eliminating human factors.

TODO: Окупятся ли затраты на проект сэкономленными деньгами?


## 2. Metrics

### Online Metrics

- На какие онлайн метрики мы хотим повлиять? (Time-To-Merge, First-Time Approval Rate, Change Failure Rate)
- Как онлайн метрики коррелируют с деньгами? Будет плюсом, если сможем свести value проекта к деньгам (наверное FTE)
- ...

### Offline Metrics

- Как будем мерить качество модели? (Precision > Recall, хотим меньше FP)
- Если модель будет писать комментарии для Review, как их будем оценивать? (LLM-as-a-judge?)
- ...

## 3. Dataset

- Какие источники данных у нас будут? Хватит ли оригинального gh-archive для извлечения информации по PR, или нужно будет использовать GH-API, чтобы обогащать данные?
- Какая таксономия изменений в коде? (пока что вижу 5: modification, creation, deletion, non-code files, documentation)
- В каком формате будет наш датасет (колонки, размер, тип датасета, вид diff hunk'ов)? 
- Будем ли балансировать распределение в датасете?
- Будем ли использовать LLM-разметку?
- Будем ли использовать ручную разметку?


## 4. Solution

### 4.1 Baseline Solution

- Как будем имплементировать бейзлайн модель?
- Как она будет генерировать message'и к diff hunk'ам?
- Какие метрики ожидаем от бейзлайна?

### 4.2 Advanced Solution

- Какая архитектура решения?
- Сколько вычислительных ресурсов требует для обучения/инференса?
- Решение от меня: BERT для классификации линий с ошибками, 1B LLM для генерации комментов

### 4.3 Measurement

- Как будем мерить качество в оффлайне?
- Какая будет у нас Success Criteria, чтобы сказать, что решение работает и несет пользу?
- Нужно ли в репортах упоминать про A/B ()?


## 5. Integration

- Как покажем, что решение работает (формат демки)?
- Где найдем ресурсы для обучения/инференса/разметки?
- Какие будут наши final deliverables?

