#version 330 core
layout(location = 0) out vec4 FragColor;

in vec3 vNormal;

uniform vec3 uColor;
uniform vec3 uLightDir;

void main() {
    // Luz direccional simple tipo Lambert
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);
    float diff = max(dot(N, L), 0.15); // mínimo 0.15 para no quedar negro total
    vec3 col = uColor * diff;
    FragColor = vec4(col, 1.0);
}
