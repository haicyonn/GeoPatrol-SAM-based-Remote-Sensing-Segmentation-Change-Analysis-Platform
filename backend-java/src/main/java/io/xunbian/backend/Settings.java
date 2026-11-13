package io.xunbian.backend;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class Settings {
    @Value("${xunbian.storage-root}") String storageRoot;
    @Value("${xunbian.python-url}") String pythonUrl;
    @Value("${xunbian.internal-token}") String internalToken;
    @Value("${xunbian.execution-enabled:true}") boolean executionEnabled;
    @Value("${xunbian.lease-seconds:60}") int leaseSeconds;
    @Value("${xunbian.prediction-timeout-seconds:180}") int predictionTimeoutSeconds;
    @Value("${xunbian.max-questions:2}") int maxQuestions;
    @Value("${xunbian.question-queue-seconds:30}") int questionQueueSeconds;
    @Value("${xunbian.max-pending-per-user:1000}") int maxPendingPerUser;
}
