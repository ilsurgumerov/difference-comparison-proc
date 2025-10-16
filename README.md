# difference-comparison-proc

# Протокол защищённого сравнения чисел с использованием троек Бивера (MPC на PyTorch Distributed)

Этот проект реализует **протокол защищённого сравнения двух чисел** $a$ и $b$ — по сути, проверку условия $a < b$ — на основе **многопартийных вычислений (MPC)**. Для координации сторон используется простой **Trusted Third Party (TTP)**, который генерирует и рассылает **тройки Бивера**, используемые в побитовых операциях `AND`.

Проект полностью реализован на **PyTorch Distributed**, без фреймворков. Взаимодействие происходит между процессами, запущенными в отдельных Docker-контейнерах.

***

## Структура проекта

```
.
├── docker-compose.yml     # Контейнеры трех ролей: ttp, worker1, worker2
├── Dockerfile             # Базовый образ, установка зависимостей
├── requirements.txt       # Python-зависимости
├── worker.py              # Универсальный запуск задач по флагам (--target)
└── tasks/
    ├── mpc_compare.py     # Основной протокол MPC сравнения
    ├── ttp.py             # Trusted Third Party (генератор троек Бивера)
    ├── auxiliary_functions.py  # Базовые операции MPC (AND, XOR, bit decomposition)
    ├── common.py          # Инициализация distributed среды
    ├── state.py           # Контейнер для хранения состояния участника
```


***

## Как это работает

**Общая идея:**

- Два участника (Worker1, Worker2) хотят сравнить свои закрытые значения `a` и `b`.
- Они не хотят раскрывать их друг другу.
- Trusted Third Party (TTP) генерирует тройки Бивера $(a,b,c)$ для побитового `AND`.
- Каждый участник получает свою долю троек $(a_i, b_i, c_i)$ и выполняет локальные побитовые операции.
- Побитово вычисляется $z = a - b$, а затем результат сравнения $z < 0$ сообщает, какое из чисел больше.

**Основные роли:**

- **TTP (rank=0)** — служба, генерирующая троики Бивера и отправляющая shares работникам.
- **Worker1 (rank=1)** — хранит значение `a` и выполняет локальные шаги MPC.
- **Worker2 (rank=2)** — хранит значение `b` (то есть `-b`, для взятие суммы: (a - b) = (a + (-b))) и выполняет зеркальные шаги MPC.

***

## Как запустить

### 1. Соберите Docker-образ

```bash
docker compose build
```


### 2. Запустите протокол

```bash
docker compose up
```

Это запустит три контейнера:

- `ttp` — сервер генерации троек Бивера
- `worker1` — участник с числом `A_VALUE`
- `worker2` — участник с числом `B_VALUE`

В `docker-compose.yml` можно задать произвольные значения:

```yaml
  worker1:
    environment:
      - A_VALUE=1234
      - MAX_DIGIT=16

  worker2:
    environment:
      - B_VALUE=777
      - MAX_DIGIT=16
```

`MAX_DIGIT` - разряд двойски, в который входят числа A и B, оба должны быть меньше 2 ** MAX_DIGIT

***

## Пример вывода в консоль

При успешном запуске вы увидите последовательность логов. Логи отдельно по воркерам для `A_VALUE=24325, B_VALUE=5465, MAX_DIGIT=16`:

**rank=0(ttp генератор троек бивера)**:
```
PS C:\Users\User\VsCodeProjects\course-confidential-AI\difference-comparison-proc> docker compose logs ttp    
ttp-1  | [Gloo] Rank 0 is connected to 2 peer ranks. Expected number of connected peer ranks is : 2
ttp-1  | [rank0] init_dist doneew Config   w Enable Watch
ttp-1  | [TTP] Beaver triple generator started
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=17)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] Generated Beaver triple (len=1)
ttp-1  | [TTP] Sent triple shares to rank1 & rank2
ttp-1  | [TTP] Waiting for requests...
ttp-1  | [TTP] All workers disconnected — shutting down gracefully.
ttp-1  | [TTP] Process group destroyed. Exit cleanly.
```

**rank=1:**
```
PS C:\Users\User\VsCodeProjects\course-confidential-AI\difference-comparison-proc> docker compose logs worker1
worker1-1  | [Gloo] Rank 1 is connected to 2 peer ranks. Expected number of connected peer ranks is : 2
worker1-1  | [rank1] init_dist done
worker1-1  | [A] A stored = 24325
worker1-1  | d = tensor([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
worker1-1  | e = tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1])
worker1-1  | Source (a - b) = x + y result in bin form: tensor([0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
worker1-1  | a - b < 0 ? : 0
```

**rank==2**
```
PS C:\Users\User\VsCodeProjects\course-confidential-AI\difference-comparison-proc> docker compose logs worker2
worker2-1  | [Gloo] Rank 2 is connected to 2 peer ranks. Expected number of connected peer ranks is : 2
worker2-1  | [rank2] init_dist done
worker2-1  | [B] B stored = -5465
worker2-1  | d = tensor([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
worker2-1  | e = tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1])
worker2-1  | Source (a - b) = x + y result in bin form: tensor([0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
worker2-1  | a - b < 0 ? : 0
```

Здесь `tensor(0)` означает, что `a - b >= 0` (то есть `a >= b`). Если `tensor(1)`, то `a < b`.

***

## Основные зависимости

- Python 3.10+
- PyTorch (distributed, tensor)
- Docker \& docker-compose

Все зависимости описаны в `requirements.txt`.
