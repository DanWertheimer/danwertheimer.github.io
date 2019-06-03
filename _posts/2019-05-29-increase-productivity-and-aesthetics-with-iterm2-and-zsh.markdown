---
title: Increasing Productivity and Aesthetics with iTerm2 and zsh
layout: post
author: Daniel Wertheimer
comments: true
tags: terminal, iTerm2, zsh
permalink: /increase-productivity-and-aesthetics-with-iterm2-and-zsh
---
# Why I Spent a lot of Time Configuring my Terminal
{% include image.html
            img="img/posts/zshIterm/HelloWorld.jpg"
            title="iTerm2 + ZSH FTW!"%}

A lot of the work I do involves tinkering around in the terminal, SSHing onto servers, managing conda environments and much more. However the amount of inefficiency in navigating the terminal bugged me. The lack of various hotkeys and shortcuts for long domain names that I resigned myself to typing in daily because it's just "what I do" was become a bit much. So, I went down the rabbit hole that is dotfile configuration and stumbled across zsh. zsh is a UNIX shell, just like terminal on your mac is a shell, however it's customization is endless. Out of the box, zsh isn't configured for anyone really but thanks to open source tools and community driven projects, [oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh) was created as a framework for managing zsh configurations. And this is where I started my journey.

# My zsh Setup