# Database Module

A comprehensive, enterprise-grade database abstraction layer for the Chatbot Framework. This module provides a unified interface for multiple database backends with built-in security, caching, and audit logging capabilities.

## Features

### 🔄 Multi-Backend Support
- **Firestore**: Google Cloud Firestore for serverless NoSQL
- **PostgreSQL**: Relational database with ACID compliance
- **Extensible**: Easy to add new database backends

### 🔒 Enterprise Security
- **Field-Level Encryption**: AES-256 encryption for sensitive data
- **Record-Level Encryption**: Full record encryption for confidential documents
- **SSL/TLS Support**: Secure connections to database servers
- **Access Control**: Built-in validation and sanitization

### ⚡ Performance Optimization
- **Redis Caching**: Intelligent caching with TTL management
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: Built-in query performance monitoring
- **Async Operations**: Non-blocking database operations

### 📊 Comprehensive Auditing
- **Operation Logging**: All database operations are logged
- **Security Events**: Unauthorized access attempts and anomalies
- **Performance Metrics**: Query execution times and cache hit rates
- **Trace IDs**: Request tracing for debugging

### 🏗️ Clean Architecture
- **Abstract Base Class**: Consistent interface across backends
- **Dependency Injection**: Modular component integration
- **Factory Pattern**: Dynamic backend selection
- **Type Safety**: Full type hints and validation

## Quick Start

### Basic Usage

```python
from database.factory import DatabaseFactory
from database.base import DatabaseConfig

# Create configuration
config = DatabaseConfig(
    backend="postgresql",
    host="localhost",
    port=5432,
    database_name="chatcore",
    username="your_username",
    password="your_password"
)

# Create database instance
factory = DatabaseFactory()
database = factory.create_database(config)

# Connect and use
await database.connect()

# Create a record
user_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "password": "secret123"  # Will be encrypted automatically
}
user_id = await database.create("users", user_data)

# Read a record
user = await database.get("users", user_id)

# Query records
results = await database.query(
    "users",
    filters={"status": "active"},
    sort_by="created_at",
    limit=10
)

# Update a record
await database.update("users", user_id, {"status": "verified"})

# Delete a record
await database.delete("users", user_id)
```

### Complete Stack with Security

```python
from database.factory import DatabaseFactory

# Load configuration from file
factory = DatabaseFactory()
config = factory.load_config("config/database.yaml")

# Create complete stack with all components
database = factory.create_complete_stack(
    config,
    encryption_key="your-32-char-encryption-key-here",
    cache_config={
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "default_ttl": 300
    },
    audit_config={
        "log_level": "INFO",
        "enable_console": True,
        "enable_file": True,
        "log_file": "logs/database_audit.log"
    }
)

await database.connect()

# All operations now include encryption, caching, and auditing
user_id = await database.create("users", sensitive_user_data)
```

## Configuration

### Database Configuration File (YAML)

```yaml
database:
  # Backend selection: firestore, postgresql
  backend: postgresql
  
  # Connection settings
  host: localhost
  port: 5432
  database_name: chatcore
  username: ${DB_USERNAME}  # Environment variable
  password: ${DB_PASSWORD}  # Environment variable
  
  # Connection pool settings
  connection_pool_size: 10
  query_timeout: 30.0
  
  # Security settings
  use_ssl: true
  ssl_cert_path: /path/to/cert.pem
  
  # Backend-specific configuration
  backend_config:
    schema: public  # PostgreSQL schema
    # project_id: your-gcp-project  # For Firestore

# Encryption configuration
encryption:
  key: ${ENCRYPTION_KEY}  # 32-character key
  sensitive_fields:
    - password
    - ssn
    - credit_card
    - api_key
  
# Cache configuration
cache:
  host: localhost
  port: 6379
  db: 0
  default_ttl: 300
  max_connections: 10
  
# Audit configuration
audit:
  log_level: INFO
  enable_console: true
  enable_file: true
  log_file: logs/database_audit.log
  buffer_size: 100
  flush_interval: 30
```

### Environment Variables

```bash
# Database credentials
export DB_USERNAME="your_db_user"
export DB_PASSWORD="your_db_password"

# Encryption key (32 characters)
export ENCRYPTION_KEY="your-32-character-encryption-key"

# Redis configuration
export REDIS_URL="redis://localhost:6379/0"

# Google Cloud (for Firestore)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

## Database Backends

### PostgreSQL

```python
config = DatabaseConfig(
    backend="postgresql",
    host="localhost",
    port=5432,
    database_name="chatcore",
    username="postgres",
    password="password",
    use_ssl=True,
    backend_config={
        "schema": "public"
    }
)
```

**Features:**
- ACID compliance
- Complex queries with joins
- Full-text search
- JSON column support
- Automatic schema management

### Firestore

```python
config = DatabaseConfig(
    backend="firestore",
    database_name="chatcore",
    backend_config={
        "project_id": "your-gcp-project-id",
        "credentials_path": "/path/to/service-account.json"
    }
)
```

**Features:**
- Serverless and auto-scaling
- Real-time updates
- Offline support
- Global distribution
- Automatic backups

## Security Features

### Data Encryption

The module automatically encrypts sensitive data before storage:

```python
# Field-level encryption (automatic)
user_data = {
    "name": "John Doe",           # Not encrypted
    "email": "john@example.com",  # Not encrypted
    "password": "secret123",      # Encrypted automatically
    "ssn": "123-45-6789"         # Encrypted automatically
}

# Record-level encryption (manual)
sensitive_document = {
    "content": "confidential information",
    "classification": "top-secret"
}

encrypted = await encryption_manager.encrypt_record(sensitive_document)
```

### Access Control

```python
# Input validation
database.validate_collection_name("users")      # ✓ Valid
database.validate_collection_name("user data")  # ✗ Invalid (spaces)

database.validate_document_id("user123")        # ✓ Valid
database.validate_document_id("user@123")       # ✗ Invalid (special chars)
```

## Caching Strategy

### Automatic Caching

```python
# Cache configuration
cache_config = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "default_ttl": 300,  # 5 minutes
    "max_connections": 10
}

# Cache keys are automatically generated
# Format: "record:{collection}:{document_id}"
# Example: "record:users:user123"

# Query cache keys include filters and pagination
# Format: "query:{collection}:{hash_of_parameters}"
```

### Cache Management

```python
# Manual cache operations
await cache_manager.set("custom_key", data, ttl=600)
cached_data = await cache_manager.get("custom_key")
await cache_manager.delete("custom_key")

# Pattern-based invalidation
await cache_manager.invalidate_pattern("user:*")
await cache_manager.invalidate_collection("users")

# Cache metrics
metrics = cache_manager.get_metrics()
print(f"Hit rate: {metrics['hit_rate']:.2%}")
```

## Audit Logging

### Automatic Logging

All database operations are automatically logged with:
- Operation type (CREATE, READ, UPDATE, DELETE, QUERY)
- Collection/table name
- Document/record ID
- Execution time
- Success/failure status
- Trace ID for request correlation

### Security Events

```python
# Automatic security event logging
await audit_logger.log_security_event(
    event_type="unauthorized_access",
    severity="HIGH",
    details={
        "user_ip": "192.168.1.100",
        "attempted_action": "admin_access",
        "user_id": "suspicious_user"
    }
)
```

### Log Analysis

```python
# Query audit logs
recent_operations = await audit_logger.get_recent_operations(
    hours=24,
    operation_types=[OperationType.CREATE, OperationType.DELETE]
)

# Performance analysis
slow_queries = await audit_logger.get_slow_queries(
    min_execution_time=1000  # milliseconds
)
```

## Performance Features

### Connection Pooling

```python
# Automatic connection pooling
config = DatabaseConfig(
    backend="postgresql",
    connection_pool_size=20,  # Max connections
    query_timeout=30.0        # Query timeout in seconds
)
```

### Bulk Operations

```python
# Bulk insert (efficient batch processing)
users = [
    {"name": "User 1", "email": "user1@example.com"},
    {"name": "User 2", "email": "user2@example.com"},
    {"name": "User 3", "email": "user3@example.com"}
]
user_ids = await database.bulk_insert("users", users)

# Bulk update
updates = [
    {"id": "user1", "data": {"status": "active"}},
    {"id": "user2", "data": {"status": "inactive"}}
]
updated_count = await database.bulk_update("users", updates)

# Bulk delete
deleted_count = await database.bulk_delete("users", {"status": "inactive"})
```

### Query Optimization

```python
# Efficient querying with pagination
result = await database.query(
    "users",
    filters={"status": "active", "age": {">=": 18}},
    sort_by="created_at",
    sort_desc=True,
    limit=50,
    offset=0
)

print(f"Total users: {result.total_count}")
print(f"Current page: {result.page}")
print(f"Has more: {result.has_more}")
print(f"Query time: {result.execution_time_ms:.2f}ms")
```

## Error Handling

### Exception Types

```python
from database.base import DatabaseError, ConnectionError, QueryError, ValidationError

try:
    await database.create("users", invalid_data)
except ValidationError as e:
    print(f"Validation failed: {e}")
except QueryError as e:
    print(f"Query failed: {e}")
    print(f"Trace ID: {e.query_id}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except DatabaseError as e:
    print(f"Database error: {e}")
```

### Health Monitoring

```python
# Health check
health = await database.health_check()
print(f"Status: {health['status']}")
print(f"Response time: {health['response_time_ms']}ms")

if health['status'] != 'healthy':
    # Handle unhealthy database
    await database.disconnect()
    await database.connect()  # Retry connection
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest chatcore/database/test_database.py -v

# Run specific test categories
python -m pytest chatcore/database/test_database.py::TestEncryptionManager -v
python -m pytest chatcore/database/test_database.py::TestCacheManager -v
python -m pytest chatcore/database/test_database.py::TestDatabaseFactory -v

# Run with coverage
pip install pytest-cov
python -m pytest chatcore/database/test_database.py --cov=chatcore.database --cov-report=html
```

### Mock Testing

```python
# Using the built-in mock database for testing
from database.test_database import MockDatabaseImplementation

config = DatabaseConfig(backend="mock", database_name="test")
test_db = MockDatabaseImplementation(config)

await test_db.connect()
user_id = await test_db.create("users", {"name": "Test User"})
user = await test_db.get("users", user_id)
```

## Extending the Module

### Adding a New Backend

1. **Create Implementation Class**:

```python
from database.base import BaseDatabase

class MyDatabaseImplementation(BaseDatabase):
    async def connect(self) -> None:
        # Implementation
        pass
    
    async def create(self, collection: str, data: DatabaseRecord, document_id: Optional[str] = None) -> str:
        # Implementation
        pass
    
    # ... implement all abstract methods
```

2. **Register with Factory**:

```python
from database.factory import DatabaseFactory

factory = DatabaseFactory()
factory.register_backend("my_database", MyDatabaseImplementation)
```

3. **Use New Backend**:

```python
config = DatabaseConfig(
    backend="my_database",
    database_name="test",
    backend_config={
        "custom_setting": "value"
    }
)
database = factory.create_database(config)
```

### Custom Encryption

```python
from database.encryption import EncryptionManager

class CustomEncryptionManager(EncryptionManager):
    async def encrypt_data(self, data: DatabaseRecord) -> DatabaseRecord:
        # Custom encryption logic
        return await super().encrypt_data(data)
```

### Custom Cache Strategy

```python
from database.cache import CacheManager

class CustomCacheManager(CacheManager):
    async def cache_query_result(self, collection: str, result: QueryResult, **kwargs):
        # Custom caching logic
        await super().cache_query_result(collection, result, **kwargs)
```

## Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY chatcore/ ./chatcore/
COPY config/ ./config/

CMD ["python", "-m", "chatcore.main"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  chatbot:
    build: .
    environment:
      - DB_USERNAME=postgres
      - DB_PASSWORD=password
      - ENCRYPTION_KEY=your-32-character-encryption-key
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=chatcore
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot-database
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chatbot-database
  template:
    metadata:
      labels:
        app: chatbot-database
    spec:
      containers:
      - name: chatbot
        image: chatbot:latest
        env:
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: encryption-key
              key: key
```

### Monitoring

```python
# Custom monitoring integration
import prometheus_client

# Database metrics
database_operations = prometheus_client.Counter(
    'database_operations_total',
    'Total database operations',
    ['backend', 'operation', 'collection']
)

database_errors = prometheus_client.Counter(
    'database_errors_total',
    'Total database errors',
    ['backend', 'error_type']
)

query_duration = prometheus_client.Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['backend', 'collection']
)
```

## Troubleshooting

### Common Issues

1. **Connection Failures**:
   - Check database server is running
   - Verify credentials and connection string
   - Check firewall and network connectivity
   - Review SSL/TLS configuration

2. **Performance Issues**:
   - Monitor cache hit rates
   - Check query execution times in audit logs
   - Review connection pool settings
   - Analyze database query plans

3. **Encryption Errors**:
   - Verify encryption key length (32 characters)
   - Check sensitive field configuration
   - Review encryption manager initialization

4. **Cache Issues**:
   - Check Redis server connectivity
   - Monitor Redis memory usage
   - Review cache TTL settings
   - Check cache invalidation patterns

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Database-specific logging
logger = logging.getLogger('database')
logger.setLevel(logging.DEBUG)
```

### Health Monitoring

```python
# Comprehensive health check
async def check_system_health():
    health_results = {}
    
    # Database health
    db_health = await database.health_check()
    health_results['database'] = db_health
    
    # Cache health
    if cache_manager:
        cache_health = await cache_manager.health_check()
        health_results['cache'] = cache_health
    
    # Overall status
    overall_status = all(
        result.get('status') == 'healthy' 
        for result in health_results.values()
    )
    
    return {
        'status': 'healthy' if overall_status else 'unhealthy',
        'components': health_results,
        'timestamp': datetime.utcnow().isoformat()
    }
```

## License

This database module is part of the Chatbot Framework and follows the same licensing terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For detailed contribution guidelines, see the main project README.
