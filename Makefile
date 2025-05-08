# Git Makefile Helper
# 使用: make <target> MSG="your message"

# 默认提交信息
MSG ?= "update"

# 添加所有修改
add:
	git add .

# 提交更改
commit: add
	git commit -m "$(MSG)"

# 添加、提交并推送
push: commit
	git push

# 拉取远程更改
pull:
	git pull --rebase

# 查看当前状态
status:
	git status

# 查看本地与远程差异
diff:
	git fetch
	git diff HEAD origin/`git rev-parse --abbrev-ref HEAD`

# 检查本地与远程的同步状态
check-sync:
	@LOCAL=$$(git rev-parse @); \
	REMOTE=$$(git rev-parse @{u}); \
	BASE=$$(git merge-base @ @{u}); \
	if [ "$$LOCAL" = "$$REMOTE" ]; then \
		echo "✅ 本地和远程完全同步"; \
	elif [ "$$LOCAL" = "$$BASE" ]; then \
		echo "⬇️ 远程有更新，需执行 pull"; \
	elif [ "$$REMOTE" = "$$BASE" ]; then \
		echo "⬆️ 本地有未推送的提交，需执行 push"; \
	else \
		echo "⚠️ 本地和远程都有更新，可能产生冲突！建议先备份或手动处理"; \
	fi

# 一键同步：先 pull --rebase，再 push
safe-update:
	@$(MAKE) check-sync
	@read -p "🔁 是否继续执行 pull --rebase 并 push? (y/n): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		git add .; \
		git commit -m "$(MSG)"; \
		git pull --rebase; \
		git push; \
	else \
		echo "❌ 操作取消"; \
	fi

# 全自动同步（不询问，适合自动化）
sync:
	git add .
	git commit -m "$(MSG)" || echo "ℹ️ 没有需要提交的内容"
	git pull --rebase
	git push
