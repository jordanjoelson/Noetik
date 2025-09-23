# Commits

This repo uses [semantic-release](https://github.com/semantic-release/semantic-release) and [angular style commits](https://github.com/angular/angular/blob/68a6a07/CONTRIBUTING.md#commit).

|build|no release|
|ci|no release|
|docs|no release|
|feat|minor|
|fix|patch|
|perf|patch|
|refactor|no release|
|test|no release|
|chore|patch|
|style|no release|

**Breaking changes result in a major release.**

## Example angular-style commits

### Non-Breaking Behavior Changes (minor/patch)

```
feat: add copyright notice to footer
```

```
fix: resolve issue with dropdown menu in navbar

- was using position: relative instead of position: absolute
- cleaned up CSS in component
```

```
perf: getUserData() route is now 20% faster

- Uses memcached that is truncated out every 5 days
- Benefits power users
```

### Breaking Changes (major)

```
feat: removed deprecated user auth endpoint

BREAKING CHANGE: removed deprecated user auth endpoint
```

### No behavior changes (no release)

```
docs: update installation instructions
```

```
style: adjust padding for primary button
```
