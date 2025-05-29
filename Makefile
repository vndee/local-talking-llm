PHONY: hello

RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
MAGENTA=\033[0;35m
CYAN=\033[0;36m
RESET=\033[0m

hello:
	@echo "${MAGENTA}Hello, $$(whoami)!${RESET}"
	@echo "${GREEN}Current Time:${RESET}\t\t${YELLOW}$$(date)${RESET}"
	@echo "${GREEN}Working Directory:${RESET}\t${YELLOW}$$(pwd)${RESET}"
	@echo "${GREEN}Shell:${RESET}\t\t\t${YELLOW}$$(echo $$SHELL)${RESET}"
	@echo "${GREEN}Terminal:${RESET}\t\t${YELLOW}$$(echo $$TERM)${RESET}"


lint:
	@echo "Running linter..."
	@source .venv/bin/activate && pre-commit run --all-files
	@echo "Done."
