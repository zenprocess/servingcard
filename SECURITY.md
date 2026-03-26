# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it via GitHub private vulnerability reporting on this repository.

Do not open a public issue for security vulnerabilities.

## Scope

ServingCard configs are YAML files describing serving parameters. They do not contain executable code. However, `servingcard apply --execute` runs shell commands generated from config files. Only apply configs from trusted sources.
