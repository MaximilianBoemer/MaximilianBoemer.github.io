---
title: Jotter
subtitle: I write notes about various things I learn about, hoping to organize my thoughts and share them with others.
layout: page
hero_image: /Road_trip_Europe-309_sauna_adapted.jpg
---

{% for post in site.posts limit:1 %}
<section class="section">
    <div class="columns is-multiline">
        <div class="column is-12">
            <div class="content">
                <h2 class="has-text-centered">Latest Post</h2>
            </div>
        </div>
        <div class="column is-12">
            <section class="hero is-dark">
                <div class="hero-body">
                    <div class="columns">
                        <div class="column is-4">
                            <img src="{{ post.image | replace: 'http:', 'https:' }}" alt="{{ post.title }}" />
                        </div>
                        <div class="column is-8">
                            <div class="has-text-centered">
                                <a href="{{ post.url | prepend: site.baseurl }}">
                                    <h3 class="title is-4">{{ post.title }}</h3>
                                </a>
                                <p class="subtitle is-6">Published: {{ post.date | date: "%b %-d, %Y" }}</p>
                                <div class="content">
                                    <p>{{ post.excerpt }}</p>
                                </div>
                                <a href="{{ post.url | prepend: site.baseurl }}" class="button is-info">
                                    Read more...
                                </a>
                            </div> 
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </div>
</section>
{% endfor %}