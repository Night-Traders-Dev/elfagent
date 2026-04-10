from core.config import ENABLE_CONSOLE_OTEL

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    HAS_OTEL = True
except ImportError:
    trace = None
    metrics = None
    Resource = None
    SERVICE_NAME = None
    TracerProvider = None
    ConsoleSpanExporter = None
    BatchSpanProcessor = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    ConsoleMetricExporter = None
    HAS_OTEL = False


def setup_otel(service_name: str = "elfagentplus"):
    if not HAS_OTEL:
        return None, None
    resource = Resource.create(attributes={SERVICE_NAME: service_name})
    tracer_provider = TracerProvider(resource=resource)
    if ENABLE_CONSOLE_OTEL:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    if ENABLE_CONSOLE_OTEL:
        meter_provider = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())])
    else:
        meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(meter_provider)
    return trace.get_tracer(service_name), metrics.get_meter(service_name)
