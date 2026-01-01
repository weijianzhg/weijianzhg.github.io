require 'feedjira'
require 'httparty'
require 'jekyll'

module ExternalPosts
  class ExternalPostsGenerator < Jekyll::Generator
    safe true
    priority :high

    def generate(site)
      if site.config['external_sources'] != nil
        site.config['external_sources'].each do |src|
          p "Fetching external posts from #{src['name']}:"

          # Use cached feed if available, otherwise fetch from URL
          cache_file = File.join(site.source, '_cache', "#{src['name']}_feed.xml")

          if File.exist?(cache_file)
            p "...using cached feed"
            xml = File.read(cache_file)
          else
            p "...fetching from URL"
            xml = HTTParty.get(src['rss_url'], headers: {'User-Agent' => 'Mozilla/5.0'}).body

            # Save to cache
            FileUtils.mkdir_p(File.dirname(cache_file))
            File.write(cache_file, xml)
            p "...cached for future builds"
          end

          feed = Feedjira.parse(xml)
          feed.entries.each do |e|
            p "...fetching #{e.url}"
            slug = e.title.downcase.strip.gsub(' ', '-').gsub(/[^\w-]/, '')
            path = site.in_source_dir("_posts/#{slug}.md")
            doc = Jekyll::Document.new(
              path, { :site => site, :collection => site.collections['posts'] }
            )
            doc.data['external_source'] = src['name'];
            doc.data['feed_content'] = e.content;
            doc.data['title'] = "#{e.title}";
            doc.data['description'] = e.summary;
            doc.data['date'] = e.published;
            doc.data['redirect'] = e.url;
            site.collections['posts'].docs << doc
          end
        end
      end
    end
  end

end
